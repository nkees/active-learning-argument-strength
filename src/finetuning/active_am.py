import tqdm as tqdm
import numpy as np
import pandas as pd
import os

from src.finetuning.run_am import predict
from src.finetuning.utils_am import load_and_cache_examples

from src.finetuning.acquisition_functions import acquisition_functions


def convert_logits_to_softmax(logits):
    return (np.exp(logits.T) / np.sum(np.exp(logits.T), axis=0)).T


def infer(args, task, tokenizer, acquisition_model, logger):
    """
    Conducts inference and prepares the data acquisition step. Saves the data in a file.
    Args:
        args: args
        task: AP or AR
        tokenizer: tokenizer
        acquisition_model: model for conducting inference
        logger: logger

    """
    logger.info("Starting inference!")
    acquisition_dataset, indexes = load_and_cache_examples(
        args, task, tokenizer, evaluate=False, validate=False, acquire=True
    )
    predictions = []
    for i in tqdm.tqdm(range(args.inference_iterations)):
        logger.info(f"Predicting for {i}th round out of {args.inference_iterations}.")
        results, _ = predict(
            args,
            logger,
            acquisition_dataset,
            acquisition_model,
            "Acquiring",
            acquire=True,
        )
        logger.info(f"len(results): {len(results)}")
        predictions.append(results)

    logits = np.array(predictions)
    softmax_outputs = convert_logits_to_softmax(logits)

    if args.acq_func not in acquisition_functions:
        raise NotImplementedError(
            f"Acquisition function {args.acq_func} is not available."
        )
    acquisition_function = acquisition_functions[args.acq_func]
    uncertainties = acquisition_function.calculate_uncertainty(softmax_outputs)

    df_res = pd.DataFrame(
        list(zip(indexes, uncertainties)), columns=["id", args.acq_func]
    )
    logger.info(f"df_results: {df_res.head()}")
    df_pool = pd.read_csv(
        os.path.join(args.output_data_dir, "remaining_training_pool.csv")
    )
    logger.info(f"df_pool: {df_pool.head()}")
    df_with_uncertainty = pd.merge(df_pool, df_res, how="left", on="id")
    logger.info(f"df_with_uncertainty: {df_with_uncertainty.head()}")
    df_with_uncertainty = df_with_uncertainty.sort_values(
        by=[args.acq_func], ascending=False
    )
    df_with_uncertainty.to_csv(
        os.path.join(args.output_data_dir, "uncertain_data.csv"), index=False
    )

