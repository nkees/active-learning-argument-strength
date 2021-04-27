import argparse
import pandas as pd
import numpy as np
import logging
import os
import re
import torch
from mlflow import log_param, log_metric
import mlflow
from datetime import datetime
from transformers import (
    WEIGHTS_NAME,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)
from src.finetuning.args import parser_arguments
from src.finetuning.utils_am import (
    load_and_cache_examples,
    output_modes,
    processors,
)
from src.finetuning.run_am import train, evaluate
from src.finetuning.active_am import infer
from src.postprocessing.clean_up import find_newest_directory, clear_directory
from src.finetuning.graph_structure import Graph
from src.postprocessing.clean_up import delete_models_in_finished_process
from checking_data import check_integrity_of_data_pool
from src.visualization.calculate_performance import report_performance_passive, find_metric
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    all_models = sum(
        (
            tuple(conf.pretrained_config_archive_map.keys())
            for conf in (
                BertConfig,
                XLNetConfig,
                XLMConfig,
                RobertaConfig,
                DistilBertConfig,
                AlbertConfig,
                XLMRobertaConfig,
            )
        ),
        (),
    )
    model_classes = {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
        "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        "distilbert": (
            DistilBertConfig,
            DistilBertForSequenceClassification,
            DistilBertTokenizer,
        ),
        "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        "xlmroberta": (
            XLMRobertaConfig,
            XLMRobertaForSequenceClassification,
            XLMRobertaTokenizer,
        ),
    }

    parser = argparse.ArgumentParser()
    parser_arguments(all_models, model_classes, parser)
    args = parser.parse_args()
    now = datetime.now()
    now = "{}-{}-{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute)

    # Check if all the arguments have been provided correctly
    if args.active_retrain and not args.active_learning:
        raise AssertionError(
            "Learning with reusing the model is not possible in any other mode than while "
            "active learning. Please make sure --args.active_learning is on."
        )
    if args.continue_learning and not args.active_learning:
        raise AssertionError(
            "Continue learning is not possible in any other mode than while active learning. Please "
            "make sure --args.active_learning is on or skip this parameter to train a new model."
        )
    if args.active_learning and not args.do_train:
        raise AssertionError(
            "Active learning is not possible in any other mode than while training. Please make sure"
            "--args.do_train is on."
        )
    if args.acq_func != "random":
        logger.info(
            f"There will be {args.inference_iterations} iterations for inference. If you want to use some "
            f"other specific number, set --inference_iterations to the amount of your choice."
        )
    model_name = os.path.split(os.path.split(args.output_dir)[0])[1]
    if args.active_learning:
        mlflow.set_experiment(f"{model_name}_argument_bert_early_active")
    else:
        mlflow.set_experiment(f"argument_bert_early")

    args.valid = False

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    regex = re.compile(r"\d+")
    number_model = int(regex.findall(args.output_dir)[0])
    logging.basicConfig(
        filename=f"logging_model_{model_name}_{now}.log",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    config_class, model_class, tokenizer_class = model_classes[args.model_type]

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # set_seed(args)
    args.model_type = args.model_type.lower()
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    OUTPUT_DIR_ORIGINAL = args.output_dir  # models/experiment_n/
    regex = re.compile(r"\d+")
    if not os.path.exists(OUTPUT_DIR_ORIGINAL):
        os.makedirs(OUTPUT_DIR_ORIGINAL)
        n = 1
    else:
        # if the existing path already has some models in it, just pick the next unused number and create a new dir
        remaining_models, newest_model_path = find_newest_directory(OUTPUT_DIR_ORIGINAL)
        n = int(regex.findall(newest_model_path)[-1])
        if not args.continue_learning:
            n += 1
    number_of_samples = args.sample_models
    if not os.path.exists(args.mask_path):
        args.mask = None
    else:
        args.mask = np.load(args.mask_path)

    if args.active_learning and args.stopping_criterion == "performance reached":
        # Reference "passive" model
        mean, _, _ = report_performance_passive(args.passive_path)
        reference_performance = float(mean)  # ToDo: in order to capture significantly higher performance, place the measure onto the upper limit
        logger.info(f"Reference performance is: {reference_performance} of type {type(reference_performance)}")

    # Define stopping condition for the active learning process: when all the data in the data pool has been used
    while args.sample_models > 0:
        OUTPUT_DIR = os.path.join(OUTPUT_DIR_ORIGINAL, f"model{n}")  # models/experiment_n/model_k
        args.output_dir = OUTPUT_DIR
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info(f"Training model {n}/{number_of_samples}. Path: {OUTPUT_DIR}.")
        learning_iteration = 0

        if args.active_learning:
            args.directed_argument_graph = Graph()
            if args.continue_learning:
                # ToDo: fix the problem (see below)
                print(
                    "Attention: Error danger! If you have deleted some of the models by hand and wish to "
                    "continue training from some specific point, it is currently impossible because "
                    "the training data pool does not correspond to this stage. This will be fixed later."
                )
                if os.path.exists(
                    os.path.join(
                        args.output_dir, "active_data", "remaining_training_pool.csv"
                    )
                ):
                    logger.info("The learning will continue.")
                    # will continue to train the newest model in the directory
                    (
                        iteration_paths,
                        latest_learning_iteration_path,
                    ) = find_newest_directory(args.output_dir)
                    learning_iteration = int(
                        regex.findall(latest_learning_iteration_path)[-1]
                    )
                    previous_training_data = pd.read_csv(os.path.join(args.output_dir, "active_data", "training_subset.csv"))
                    args.directed_argument_graph.push_argument_pairs_from_df_to_graph(previous_training_data)

            args.output_data_dir = os.path.join(
                OUTPUT_DIR, "active_data"
            )  # models/experiment_n/model_k/active_data
        else:
            args.output_data_dir = OUTPUT_DIR  # models/experiment_n/model_k/
        if not os.path.exists(args.output_data_dir):
            os.makedirs(args.output_data_dir)
        # move the data to the internal model folder for better traceability
        processor.prepare_train_data(
            path_from=args.data_dir, path_to=args.output_data_dir
        )

        stopping_condition = 1  # when 0, then stop learning

        # if no active learning, the model will be trained only once
        while stopping_condition > 0:
            # Use the non-finetuned model as the base, if not args.active_retrain
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )

            if args.active_learning and args.do_train:
                args.output_dir = os.path.join(
                    OUTPUT_DIR, f"learning_iteration_{learning_iteration}"
                )
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                logger.info(
                    "Learning iteration: {} of approx. {}".format(
                        learning_iteration,
                        str(
                            (9100 - 9100 * args.validation_size) # ToDo: fix the magic number!!!
                            // args.active_learning_batch_size
                        ),
                    )
                )

                # Load a model from the previous active learning round as the model for inference
                path_latest_iteration = os.path.join(
                    OUTPUT_DIR, f"learning_iteration_{learning_iteration - 1}"
                )
                if os.path.exists(
                    path_latest_iteration
                ):  # reachable if learning iteration > 0
                    try:
                        logger.info(
                            f"Using model for acquisition from: {LATEST_BEST_MODEL_PATH}"  # ToDo: fix this
                        )
                    except NameError:
                        _, LATEST_BEST_MODEL_PATH = find_newest_directory(path_latest_iteration)
                        logger.info(
                            f"Using model for acquisition from: {LATEST_BEST_MODEL_PATH}"
                        )
                    acquisition_model = model_class.from_pretrained(
                        LATEST_BEST_MODEL_PATH
                    )
                    acquisition_model.to(args.device)
                    # exists if there has been at least one learning round
                    if (
                        args.active_retrain
                    ):  # ToDo: implement continued training for active_retrain
                        model = model_class.from_pretrained(LATEST_BEST_MODEL_PATH)

            mlflow.start_run()

            if args.local_rank == 0:
                # Make sure only the first process in distributed training will download model & vocab
                torch.distributed.barrier()

            model.to(args.device)
            logger.info("Training/evaluation parameters %s", args)
            log_param("data_dir", args.data_dir)
            log_param("model_type", args.model_type)
            log_param("model_name_or_path", args.model_name_or_path)
            log_param("task_name", args.task_name)
            log_param("output_dir", args.output_dir)
            if args.active_learning:
                log_param("output_data_dir", args.output_data_dir)

            # Training
            if args.do_train:
                # Inference
                if args.active_learning:
                    if (
                        learning_iteration > 0
                        and args.acq_func != "random"
                        and not args.continue_learning
                    ):
                        # no inference at the very first round or when random acquisition
                        # ToDo: implement no inference for continue_learning if cached file is not there yet
                        infer(
                            args, args.task_name, tokenizer, acquisition_model, logger
                        )
                # Load data
                train_dataset = load_and_cache_examples(
                    args, args.task_name, tokenizer, evaluate=False, validate=False
                )
                # Returns the best model on validation set
                model, global_step, OUTPUT_DIR_BEST = train_prep(
                    args, model, model_class, tokenizer, train_dataset
                )

            # Evaluation - depending whether after training or just evaluating
            if args.do_train and args.do_eval and args.local_rank in [-1, 0]:
                eval_prep(args, model, tokenizer, global_step)
            elif args.do_eval and args.local_rank in [-1, 0] and not args.do_train:
                eval_prep(args, model, tokenizer)

            torch.cuda.reset_max_memory_allocated(args.device)

            mlflow.end_run()

            # check if there is the data pool and if something is left there and use it as stopping condition
            if args.active_learning and args.do_train:
                # save only the best model and get rid of the rest
                clear_directory(args.output_dir, logger)

                if args.stopping_criterion == "performance reached":
                    eval_acc = float(find_metric(os.path.join(args.output_dir, "eval_results.txt"), "acc = "))
                    logger.info(f"Evaluation accuracy is: {eval_acc} of type {type(eval_acc)}")
                    logger.info(f"Is eval_acc larger as reference performance? - {eval_acc >= reference_performance}")
                    if eval_acc >= reference_performance:
                        stopping_condition = 1  # will reduced to 0 in the next step
                    else:
                        df_pool = pd.read_csv(os.path.join(args.output_data_dir, "remaining_training_pool.csv"))
                        # +1 in case there is still just one example which needs to be trained
                        stopping_condition = len(df_pool) + 1
                elif args.stopping_criterion == "empty set":
                    df_pool = pd.read_csv(os.path.join(args.output_data_dir, "remaining_training_pool.csv"))
                    # +1 in case there is still just one example which needs to be trained
                    stopping_condition = len(df_pool) + 1
                else:
                    if learning_iteration == int(args.stopping_criterion):
                        stopping_condition = 1
                    else:
                        df_pool = pd.read_csv(os.path.join(args.output_data_dir, "remaining_training_pool.csv"))
                        # +1 in case there is still just one example which needs to be trained
                        stopping_condition = len(df_pool) + 1

            stopping_condition -= 1
            learning_iteration += 1
            if args.do_train:
                LATEST_BEST_MODEL_PATH = OUTPUT_DIR_BEST
            if stopping_condition == 0:
                logger.info("Successfully completed.")
                logger.info(
                    "--------------------------------------------------------------------------------\n"
                )
            # finish the continued process and if the sampling goes on, set continue_learning to false
            args.continue_learning = False
            if args.graph_method:
                check_integrity_of_data_pool(args.output_data_dir)
        n += 1
        # delete the models after the round
        delete_models_in_finished_process(OUTPUT_DIR)

        args.sample_models -= 1
    if not args.active_learning:
        delete_models_in_finished_process(OUTPUT_DIR_ORIGINAL)


def train_prep(args, model, model_class, tokenizer, train_dataset):
    # At this point, model is the bert-base-uncased model
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        # Create output directory if needed
        bert_base_path = os.path.join(args.output_dir, "checkpoint-0")
        if not os.path.exists(bert_base_path) and args.local_rank in [-1, 0]:
            os.makedirs(bert_base_path)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(bert_base_path)
        tokenizer.save_pretrained(bert_base_path)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(bert_base_path, "training_args.bin"))
    # Train the model
    global_step, tr_loss, output_dir = train(
        args, train_dataset, model, tokenizer, logger
    )
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(output_dir)
    model.to(args.device)
    return model, global_step, output_dir


def eval_prep(args, model, tokenizer, global_step=None):
    eval_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, evaluate=True
    )
    result = evaluate(args, eval_dataset, model, logger, valid=False)
    if args.do_train:
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        for key, value in result.items():
            log_metric("test_{}".format(key), value, global_step)


if __name__ == "__main__":
    main()
