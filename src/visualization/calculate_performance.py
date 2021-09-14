import pandas as pd
import os
import re
import scipy.stats as stats
import numpy as np


def find_metric(path, keyword):
    with open(path, "r") as reader:
        lines = reader.readlines()
        for i in lines:
            if keyword in i:
                searched_metric = i.split(" = ")[1].strip()
    return float(searched_metric)


def collect_subdirs(models_dir):
    if not os.path.exists(models_dir):
        raise ValueError(
            "The directory ({}) does not exist. Please suggest a valid directory with the models and records of model performance.".format(
                models_dir
            )
        )
    # logger.info("Data for the performance visualisation will be taken from: ({}).".format(models_dir))
    subdirs = [
        f.path
        for f in os.scandir(models_dir)
        if f.is_dir() and "active_data" not in f.path
    ]
    return subdirs


def collect_multiple_model_performance(models_dir, args):
    experiment_subdirs = collect_subdirs(models_dir)
    args.size = len(experiment_subdirs)
    header = [["model_name", "num_test_data", "valid_acc", "eval_acc"]]
    column_names = header.pop(0)
    df_multiple = pd.DataFrame(header, columns=column_names)
    for i in experiment_subdirs:
        if "model18" not in i:
            df_single = collect_single_model_performance(i)
            df_single["model_name"] = i
            df_single = df_single[column_names]
            df_multiple = pd.concat([df_multiple, df_single])
    df_multiple.to_csv(f"concatenated_experiment_results_test{models_dir.split('/')[1].strip('/')}.csv", index=False)
    return df_multiple


def collect_single_model_performance(models_dir):  # , logger):
    subdirs = collect_subdirs(models_dir)
    assert "learning_iteration" in subdirs[0]
    active_learning_iterations = subdirs
    # logger.info("Collecting the data...")
    regex = re.compile(r"\d+")
    metrics = [["num_test_data", "valid_acc", "eval_acc"]]
    for i in active_learning_iterations:
        iteration_num = int(regex.findall(os.path.split(i)[1])[0])
        # logger.info("Collecting the data... {} of {} iterations.".format(iteration_num, len(active_learning_iterations)))
        model_path = [f.path for f in os.scandir(i) if f.is_dir() and "active_data" not in f.path][0]
        num_test_data_points = find_metric(
            os.path.join(i, "input_nums.txt"), "Num examples training"
        )
        valid_acc = find_metric(os.path.join(model_path, "valid_results.txt"), "acc = ")
        eval_acc = find_metric(os.path.join(i, "eval_results.txt"), "acc = ")
        metrics.append([num_test_data_points, valid_acc, eval_acc])
    column_names = metrics.pop(0)
    data = pd.DataFrame(metrics, columns=column_names)
    data = data.sort_values(by=["num_test_data"])
    return data


def collect_model_performance_passive(models_dir):
    subdirs = collect_subdirs(models_dir)
    regex = re.compile(r"\d+")
    metrics = [["model_num", "eval_acc"]]
    for i in subdirs:
        model_num = int(regex.findall(os.path.split(i)[1])[0])
        eval_acc = find_metric(os.path.join(i, "eval_results.txt"), "acc = ")
        metrics.append([model_num, eval_acc])
    column_names = metrics.pop(0)
    data = pd.DataFrame(metrics, columns=column_names)
    return data


def report_performance_passive(models_dir):
    df = collect_model_performance_passive(models_dir)
    mean_value = df["eval_acc"].mean()
    std_value = df["eval_acc"].std()

    alpha = 0.05
    subdirs = collect_subdirs(models_dir)
    size = len(subdirs)

    upper_interval = mean_value + stats.t.ppf(1.0 - (alpha / 2.0), size - 1) * (
            std_value / np.sqrt(size)
    )
    lower_interval = mean_value - stats.t.ppf(1.0 - (alpha / 2.0), size - 1) * (
            std_value / np.sqrt(size)
    )
    return mean_value, upper_interval, lower_interval
