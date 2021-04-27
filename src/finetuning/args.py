from src.finetuning.utils_am import processors as processors


def parser_arguments(all_models, model_classes, parser):
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files for the task.",
    )
    parser.add_argument(
        "--test",
        type=int,
        required=True,
        help="The topic id of the test_set. Set value -1 for random selection from all topics.",
    )
    parser.add_argument(
        "--validation_size",
        default=0.2,
        type=float,
        required=True,
        help="The size of the validation set based on the train-set",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(model_classes.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(all_models),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: "
        + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--mask_path",
        default=None,
        type=str,
        required=True,
        help="Path to the mask (numpy array) to use for a train/valid split, add for reproducibility of experiments."
        "If no mask present yet, specify the name of your future mask for your experiments"
        "E.g. masks/ibm_args.npy",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--inference_iterations",
        default=20,
        type=int,
        help="Define a number of iterations for conducting inference for acquisition function",
    )
    parser.add_argument(
        "--active_learning",
        action="store_true",
        help="Whether to run the active learning pipeline.",
    )
    parser.add_argument(
        "--acq_func",
        default="random",
        type=str,
        help="Selection strategy for active learning.",
    )
    parser.add_argument(
        "--graph_method",
        action="store_true",
        help="Whether to use boosting through graph-based methods or use just simple heuristics, when active learning.",
    )
    parser.add_argument(
        "--graph_scoring",
        action="store_true",
        help="Whether to use the graph-scoring method.",
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        help="If active learning, set the directory for the data.",
    )
    parser.add_argument(
        "--active_learning_batch_size",
        default=13,
        type=int,
        help="If active learning, define the batch size of newly acquired data for each new round.",
    )
    parser.add_argument("--stopping_criterion",
                        type=str,
                        default="empty set",
                        help='Define the stopping criterion, if doing active learning. Select between: "empty set" or "performance reached" or number of the learning iteration.')
    parser.add_argument("--passive_path",
                        type=str,
                        help="If stopping criterion is performance reached, then provide the path to your reference "
                             "models here.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="If active learning, define the data pairs selection strategy with respect to topics. 'distributed' means same amount across all topics",
    )
    parser.add_argument(
        "--active_retrain",
        action="store_true",
        help="If active learning, retrain the best model from the recent iteration",
    )
    parser.add_argument(
        "--sample_models",
        default=1,
        type=int,
        help="Train a few samples of the given model. Give the number of sample models to train.",
    )
    parser.add_argument(
        "--continue_learning",
        action="store_true",
        help="If active learning and using an existing directory, define whether you want to continue learning there.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps. If -2, log after a half of an epoch,"
        "if -1, log (validate) after every epoch.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
