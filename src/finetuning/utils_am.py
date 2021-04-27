import logging
import os

from transformers.file_utils import is_tf_available
from transformers.data.processors import InputExample, InputFeatures, DataProcessor
from torch.utils.data import TensorDataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
import torch
import pandas as pd
import random
import numpy as np

from src.finetuning.graph_structure import Graph
from src.finetuning.data_acquisition import RandomAcquisition, UncertaintyAcquisition, DiversityAcquisition, GraphDiversityAcquisition, GraphScoreAcquisition, acquisition_functions

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(
    args, task, tokenizer, evaluate=False, validate=False, acquire=False
):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training process the dataset, the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]

    # Load data features from cache or dataset file
    if validate and not evaluate and not acquire:
        cached_features_file = os.path.join(
            args.output_data_dir,
            "cached_{}_{}_{}_{}".format(
                "valid",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )

    elif evaluate and not validate and not acquire:
        cached_features_file = os.path.join(
            args.output_data_dir,
            "cached_{}_{}_{}_{}".format(
                "test",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )

    elif not evaluate and not validate and not acquire:
        # if active learning, the train data will be saved inside each learning iteration directory
        cached_features_file = os.path.join(
            args.output_dir,
            "cached_{}_{}_{}_{}".format(
                "train",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )

    if (
        not acquire
        and os.path.exists(cached_features_file)
        and not args.overwrite_cache
    ):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.output_data_dir)
        label_list = processor.get_labels()
        if validate and not evaluate and not acquire:
            examples = processor.get_valid_examples(args)
        elif evaluate and not validate and not acquire:
            examples = processor.get_test_examples(args)
        elif acquire and not evaluate and not validate:
            examples, indexes = processor.get_acquire_examples(args)
        elif not evaluate and not validate and not acquire:
            examples = processor.get_train_examples(args)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0] and not acquire:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    logger.info(f"all_input_ids: {all_input_ids}")
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    logger.info(f"all_attention_mask: {all_attention_mask}")
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
    # ToDo: Unite the returns
    if acquire and not evaluate and not validate:
        return dataset, indexes
    else:
        return dataset


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet
        where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert (
            len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert (
            len(token_type_ids) == max_length
        ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids])
            )
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                },
                tf.int64,
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


def detect_dataset_name(path):
    if "IBM" in path:
        return "IBM"
    elif "UKP" in path:
        return "UKP"

class APProcessor(DataProcessor):
    """Processor for the AM data set."""

    def get_train_examples(self, args):
        """
        Prepares the training data.
        Args:
            args: args
        Returns: examples created from the necessary (slice of the) training data
        """
        dataset_name = detect_dataset_name(args.data_dir)
        if args.active_learning:
            if args.graph_method and not args.distributed and not args.graph_scoring:
                acquisition_class = GraphDiversityAcquisition(args.acq_func, args.output_data_dir, args.directed_argument_graph)
            elif args.graph_scoring and not args.distributed and not args.graph_method:
                acquisition_class = GraphScoreAcquisition(args.acq_func, args.output_data_dir)
            elif args.distributed and not args.graph_method and not args.graph_scoring:
                acquisition_class = DiversityAcquisition(args.acq_func, args.output_data_dir)
            elif args.acq_func == "random" and not args.graph_method and not args.distributed and not args.graph_scoring:
                acquisition_class = RandomAcquisition(args.acq_func, args.output_data_dir)
            elif args.acq_func in acquisition_functions and not args.graph_method and not args.distributed and not args.graph_scoring:
                acquisition_class = UncertaintyAcquisition(args.acq_func, args.output_data_dir)
            else:
                raise NotImplementedError("This acquisition mode has not been recognized or implemented.")

            if not os.path.exists(os.path.join(args.output_data_dir, "remaining_training_pool.csv")):
                _ = self.read_and_mask_training_pool(args)
            if args.graph_method:
                df_train_set, args.directed_argument_graph = acquisition_class.get_data_subset(args.active_learning_batch_size, args.sample_models, args.test, dataset_name)
            else:
                df_train_set = acquisition_class.get_data_subset(args.active_learning_batch_size, args.sample_models, args.test, dataset_name)
        else:
            df_train_set = self.read_and_mask_training_pool(args)
        training_data, _ = self._create_examples(df_train_set)
        return training_data

    def get_test_examples(self, args):
        """
        Prepares the training data.
        Returns: examples created from the test data
        """
        df = self.read_tsv(args.data_dir)
        df_test_set = df[df["new_topic_id"] == args.test]
        test_data, _ = self._create_examples(df_test_set)
        return test_data

    def get_valid_examples(self, args):
        """
        Prepares the training data.
        Returns: examples created from the validation data
        """
        # validation is always fixed regardless of whether it is active learning or not
        df = self.read_tsv(args.data_dir)
        df_train_set = df[df["new_topic_id"] != args.test]
        if args.mask is None:
            args.mask = (
                np.random.rand(len(df_train_set)) < 1 - args.validation_size
            )  # 0.8
            np.save(args.mask_path, args.mask)
        df_valid_set = df_train_set[~args.mask]
        valid_data, _ = self._create_examples(df_valid_set)
        return valid_data

    def get_acquire_examples(self, args):
        """
        Prepares the data for the acquisition step.
        Returns: examples created for the acquisition step
        """
        df_pool = self.read_tsv(
            os.path.join(args.output_data_dir, "remaining_training_pool.csv")
        )
        df_pool, indexes = self._create_examples(df_pool)
        return df_pool, indexes

    def read_and_mask_training_pool(self, args):
        """
        Pre-processing step for the training data to exclude the validation data points from the sample
        Returns: slice of the data pool with the training data set

        """
        df = self.read_tsv(os.path.join(args.output_data_dir, "training_pool_all.csv"))
        df_train_set = df[df["new_topic_id"] != args.test]
        # ToDo implement testing on randomly distributed topics ("args.test=-1")
        if args.mask is None:
            args.mask = (
                np.random.rand(len(df_train_set)) < 1 - args.validation_size
            )  # 0.85
            np.save(args.mask_path, args.mask)
        df_train_set = df_train_set[args.mask]
        df_train_set.to_csv(os.path.join(args.output_data_dir, "training_pool_all.csv"), index=False)
        return df_train_set

    def prepare_train_data(self, path_from, path_to):
        """
        Move the data to the local model directory.
        Args:
            path_from: path where the original data set is situated
            path_to: path within the model directory where the data pool should be stored

        """
        df = self.read_tsv(path_from)
        df.to_csv(os.path.join(path_to, "training_pool_all.csv"), index=False)

    @staticmethod
    def read_tsv(input_file):
        return pd.read_csv(input_file)

    @staticmethod
    def get_labels(**kwargs):
        """See base class."""
        return ["a1", "a2"]

    @staticmethod
    def _create_examples(df):
        """Creates examples for the training and test sets."""
        examples = []
        indexes = []
        i = 0
        for index, row in df.iterrows():
            guid = i
            text_a = row["a1"]
            text_b = row["a2"]
            label = row["label"]
            id_ = row["id"]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
            indexes.append(id_)
            i = i + 1
        return examples, indexes


class ARProcessor(DataProcessor):
    """Processor for the AM data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self.read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    @staticmethod
    def read_tsv(input_file):
        return pd.read_csv(input_file, sep="\t")

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples for the training and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "train":
                text_a = line[1]
                text_b = line[3]
                label = line[6]
            else:
                text_a = line[1]
                text_b = line[3]
                label = line[6]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")
    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "acc_and_f1": (acc + f1_micro) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "AP":
        return acc_and_f1(preds, labels)
    elif task_name == "AR":
        return pearson_and_spearman(preds, labels)
    else:
        raise KeyError(task_name)


tasks_num_labels = {
    "AP": 2,
}
processors = {
    "AP": APProcessor,
    "AR": ARProcessor,
}
output_modes = {
    "AP": "classification",
    "AR": "regression",
}
