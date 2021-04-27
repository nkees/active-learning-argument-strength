import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from mlflow import log_metric, log_param
from transformers import AdamW, get_linear_schedule_with_warmup
from src.finetuning.utils_am import load_and_cache_examples, set_seed, compute_metrics
import random
import string


def randomString(stringLength=8):
    """ Returns a random string. """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))


def enable_dropout(model):
    """
    Enables dropout at test time.
    Args:
        model: model

    """
    n = 0
    for each_module in model.modules():
        if each_module.__class__.__name__.startswith("Dropout"):
            n = n + 1
            each_module.train()
    print("Number of layers with dropout enabled : " + str(n)) # 38
    # all layers : 219


def document_nums(args, description, figure):
    """
    Logs numbers of training/test/validation data points to a file.
    Args:
        args: args
        description: Training, evaluation or validation
        figure: the actual amount to log

    """
    input_nums_file = os.path.join(args.output_dir, "input_nums.txt")
    if os.path.exists(input_nums_file):
        with open(input_nums_file, "r") as reader:
            contents = reader.read()
            if description not in contents:
                with open(input_nums_file, "a") as writer:
                    writer.write("%s = %s\n" % (description, str(figure)))
            # else do nothing, because the value is already there
    else:
        with open(input_nums_file, "w") as writer:
            writer.write("%s = %s\n" % (description, str(figure)))


def document_nums(args, description, figure):
    input_nums_file = os.path.join(args.output_dir, "input_nums.txt")
    if os.path.exists(input_nums_file):
        with open(input_nums_file, "r") as reader:
            contents = reader.read()
            if description not in contents:
                with open(input_nums_file, "a") as writer:
                    writer.write("%s = %s\n" % (description, str(figure)))
            # else do nothing, because the value is already there
    else:
        with open(input_nums_file, "w") as writer:
            writer.write("%s = %s\n" % (description, str(figure)))


def train(args, train_dataset, model, tokenizer, logger):
    """ Train the model """

    best_model_path = None
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps  # total optimization testing
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Document the nums of examples
    document_nums(args, "Num examples training", len(train_dataset))

    # Train!
    log_param("  Num examples training", len(train_dataset))
    log_param("  Num Epochs training", args.num_train_epochs)
    log_param("  Instantaneous batch size per GPU", args.per_gpu_train_batch_size)
    log_param(
        "  Total train batch size",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    log_param("  Gradient Accumulation steps", args.gradient_accumulation_steps)
    log_param("  Total optimization steps", t_total)

    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    global_step = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (
            len(train_dataloader) // args.gradient_accumulation_steps
        )
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps
        )

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch",
            steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )

    valid_acc = 0
    tolerance = 0
    log_metric("tolerance", 0)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        # Validate after a half epoch if so specified (-2) or after one epoch (-1)
        if args.logging_steps == -2:
            logging_steps = len(train_dataloader) // 2
        elif args.logging_steps == -1:
            logging_steps = len(train_dataloader)
        else:
            logging_steps = args.logging_steps

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if (
                    args.local_rank in [-1, 0]
                    and logging_steps > 0
                    and global_step % logging_steps == 0
                ):
                    args.valid = True
                    validate_dataset = load_and_cache_examples(
                        args, args.task_name, tokenizer, evaluate=False, validate=True
                    )
                    results = evaluate(
                        args, validate_dataset, model, logger, valid=True
                    )
                    for key, value in results.items():
                        log_metric("eval_{}".format(key), value, global_step)
                        if key == "acc":
                            valid_acc_temp = results[key]
                    loss_scalar = (tr_loss - logging_loss) / logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    log_metric("learning_rate", learning_rate_scalar, global_step)
                    log_metric("loss", loss_scalar, global_step)
                    logging_loss = tr_loss
                    if valid_acc_temp > valid_acc:
                        # model got better on validation set
                        valid_acc = valid_acc_temp
                        tolerance = 0
                        output_dir = os.path.join(
                            args.output_dir, "checkpoint-{}".format(global_step)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Additional evaluation saving for each checkpoint
                        output_valid_file = os.path.join(
                            output_dir, "valid_results.txt"
                        )
                        with open(output_valid_file, "w") as writer:
                            for key in sorted(results.keys()):
                                writer.write("%s = %s\n" % (key, str(results[key])))
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        best_model_path = output_dir
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir
                        )

                    else:
                        tolerance = tolerance + 1
                    args.valid = False
                    if tolerance > 3:
                        log_metric("tolerance", 1)
                        epoch_iterator.close()
                        train_iterator.close()
                        logger.info(
                            "Finished at Epoch: {}".format(
                                global_step // len(train_dataloader)
                            )
                        )
                        return global_step, tr_loss / global_step, best_model_path


def predict(args, logger, eval_dataset, model, desc, acquire=False):
    """ Produces predictions for the data """
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc):
        model.eval()
        if acquire:
            enable_dropout(model)
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        # np.savetxt(os.path.join(args.output_dir, "weights_output_{}.txt".format(randomString())), logits)

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )
    eval_loss = eval_loss / nb_eval_steps

    return preds, out_label_ids


def evaluate(args, eval_dataset, model, logger, valid, acquire=False):
    """ Evaluates the data and logs model performance (accuracy etc.) """
    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    # Eval!
    if valid:
        logger.info("***** Running validation {} *****")
        log_param("  Num examples valid", len(eval_dataset))
        log_param("  Batch size valid", args.eval_batch_size)
        desc = "Validating"
        # Document the nums of examples
        document_nums(args, "Num examples valid", len(eval_dataset))
    else:
        logger.info("***** Running evaluation {} *****")
        log_param("  Num examples test", len(eval_dataset))
        log_param("  Batch size test", args.eval_batch_size)
        desc = "Evaluating"
        # Document the nums of examples
        document_nums(args, "Num examples test", len(eval_dataset))

    preds, out_label_ids = predict(
        args, logger, eval_dataset, model, desc, acquire=False
    )

    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task_name, preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))
    return results
