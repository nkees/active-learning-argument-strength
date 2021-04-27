import os
import re
import shutil
import argparse
import logging

logger = logging.getLogger(__name__)


def find_subdirectories(dir):
    subfolders = [
        f.path for f in os.scandir(dir) if f.is_dir() and "active_data" not in f.path
    ]
    return subfolders


def find_newest_directory(dir):
    subfolders = find_subdirectories(dir)
    if len(subfolders) > 1:
        regex = re.compile(r"\d+")
        list_of_numbers = []
        for i in subfolders:
            tail = os.path.split(i)[1]
            logger.info("Searching in: ({})".format(tail))
            try:
                list_of_numbers.append(int(regex.findall(tail)[0]))
            except:
                list_of_numbers.append(-1)
        logger.info("List of numbers: {}".format(list_of_numbers))
        largest = subfolders.pop(list_of_numbers.index(max(list_of_numbers)))
        # ToDo: IndexError: pop from empty list - fix this in case no folders are there
        remaining_subfolders = subfolders
    else:
        largest = subfolders.pop(0)
        remaining_subfolders = []
    return remaining_subfolders, largest


def clear_directory(dir, logger):
    remaining_subfolders, to_keep = find_newest_directory(dir)
    if to_keep is not []:
        logger.info(
            "Keeping the final model in ({}). The rest ({}) will be deleted.".format(
                to_keep, remaining_subfolders
            )
        )
        for path in remaining_subfolders:
            if os.path.exists(path):
                # removing the directories
                shutil.rmtree(path)
            else:
                # file not found message
                logger.info("File not found in the directory")
    else:
        logger.info("There is only one folder in the directory. Nothing will be done.")


def delete_model(dir):
    """
    Delete weights for a single learning iteration.
    Args:
        dir: model directory
    """
    path_model = os.path.join(dir, "pytorch_model.bin")
    path_optimizer = os.path.join(dir, "optimizer.pt")
    vocab = os.path.join(dir, "vocab.txt")
    scheduler = os.path.join(dir, "scheduler.pt")
    tokenizer_config = os.path.join(dir, "tokenizer_config.json")
    special_tokens = os.path.join(dir, "special_tokens_map.json")
    training_data = os.path.join(dir, "cached_train_bert-base-uncased_80_AP")
    valid_data = os.path.join(dir, "cached_valid_bert-base-uncased_80_AP")
    test_data = os.path.join(dir, "cached_test_bert-base-uncased_80_AP")
    training_pool_all = os.path.join(dir, "training_pool_all.csv")
    list_to_delete = [path_model, path_optimizer, vocab, scheduler, tokenizer_config, special_tokens, training_data, valid_data, test_data, training_pool_all]
    for i in list_to_delete:
        if os.path.exists(i):
            os.remove(i)


def delete_all_models(dir):
    """
    Delete all models in the given experiment consisting of x complete active learning process with y learning iterations each
    Args:
        dir: experiment directory
    """
    path_list = []
    model_iterations = find_subdirectories(dir)
    path_list.append(model_iterations)
    for round in model_iterations:
        learning_iterations = find_subdirectories(round)
        for iteration in learning_iterations:
            model_dirs = find_subdirectories(iteration)
            full_path = iteration
            for i in model_dirs:
                delete_model(i)
            print(f"Cleared the directory at '{full_path}'")


def delete_models_in_finished_process(model_path):
    """
    Delete models in one complete active learning process with x learning iterations
    Args:
        model_path: model directory
    """
    learning_iterations = find_subdirectories(model_path)
    for iteration in learning_iterations:
        model_dirs = find_subdirectories(iteration)
        full_path = iteration
        delete_model(full_path)
        for i in model_dirs:
            delete_model(i)
        print(f"Cleared the directory at '{full_path}'")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="The model data dir. Contains the models and records of the performance to visualize.",
    )
    args = parser.parse_args()
    delete_all_models(args.models_dir)


if __name__ == "__main__":
    main()
