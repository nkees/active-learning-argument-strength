import argparse
import logging
from src.postprocessing.clean_up import delete_models_in_finished_process, clear_directory, find_subdirectories

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="The model data dir. Contains the models and records of the performance to visualize.",
    )
    args = parser.parse_args()
    model_iterations = find_subdirectories(args.models_dir)
    for i in model_iterations:
        clear_directory(i, logger)
    delete_models_in_finished_process(args.models_dir)


if __name__ == "__main__":
    main()