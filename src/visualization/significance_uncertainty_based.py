import argparse
import scipy.stats as stats
from calculate_performance import (
    collect_single_model_performance,
    collect_multiple_model_performance,
    collect_model_performance_passive,
    collect_subdirs,
)
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
args = parser.parse_args()


def statistical_significance(path_list1, path_list2, metric1, metric2, num_learning_iterations=0):
    """

    Args:
        path_list1: baseline
        path_list2: other heuristic
        metric1: baseline's name
        metric2: other heuristic's name

    Returns:

    """
    df_path_list = [path_list1, path_list2]
    df_list = []
    if metric2 == "passive":
        temp = pd.DataFrame(columns=["model_name", "num_test_data", "valid_acc", "eval_acc"])
        for path_path in path_list1:
            collected = collect_multiple_model_performance(path_path, args)
            collected = collected.sort_values(by=['num_test_data'])
            print(len(collected))
            temp = pd.concat([temp, collected[270:280]])
        df_list.append(temp)
        temp = pd.DataFrame(columns=["model_name", "num_test_data", "valid_acc", "eval_acc"])
        for path_path in path_list2:
            collected_p = collect_model_performance_passive(path_path)
            temp = pd.concat([temp, collected_p[0:num_learning_iterations*10]])
        df_list.append(temp)
    else:
        if num_learning_iterations == 0:
            shortest_length_list = []
            for path in df_path_list:
                for path_path in path:
                    collected = collect_multiple_model_performance(path_path, args)
                    shortest_length_list.append(len(collected))
            shortest_length_raw = min(shortest_length_list)
            shortest_length = shortest_length_raw - shortest_length_raw % 10
        else:
            shortest_length = num_learning_iterations*10

        print(shortest_length)
        for path in df_path_list:
            temp = pd.DataFrame(columns=["model_name", "num_test_data", "valid_acc", "eval_acc"])
            for path_path in path:
                collected = collect_multiple_model_performance(path_path, args)
                collected = collected.sort_values(by=['num_test_data'])
                temp = pd.concat([temp, collected[0:shortest_length]]) # e.g. 210 for 20 learning iterations, 280 for 27
            temp = temp.sort_values(by=['num_test_data'])
            df_list.append(temp)

    df_list_new = []
    for i in range(len(df_list)):
        print(f"    Collecting in {i}...")

        df = df_list[i][:]
        df_list_new.append(df)

    rvs1 = df_list_new[0]["eval_acc"]
    mean1 = np.mean(rvs1)
    variation1 = stats.variation(rvs1)
    rvs2 = df_list_new[1]["eval_acc"]
    mean2 = np.mean(rvs2)
    variation2 = stats.variation(rvs2)
    # For illustrating the differences
    # r1 = np.array(rvs1)
    # r2 = np.array(rvs2)
    # differences = r1 - r2
    # import matplotlib.pyplot as plt
    # plt.scatter(range(len(differences)), differences)
    # plt.show()

    res = stats.wilcoxon(rvs1, rvs2)
    pvalue = res[1]
    diff = mean2 - mean1
    if pvalue <= 0.0001:
        asterisk = "****"
    elif pvalue <= 0.001:
        asterisk = "***"
    elif pvalue <= 0.01:
        asterisk = "**"
    elif pvalue <= 0.05:
        asterisk = "*"
    else:
        asterisk = ""

    return f"{metric1}, mean: {round(mean1, 4)}, variation: {round(variation1, 4)} & {metric2}, mean: {round(mean2, 4)}, variation: {round(variation2, 4)} & avg.diff.: {round(diff, 4)}{asterisk}"

final_results = []

final_results.append("IBM")

path_entropy = ["models/experiment_57_entropy_random/", "models/experiment_57_entropy_random_3/", "models/experiment_57_entropy_random_7/"]
path_var_rat = ["models/experiment_58_var_rat_random/", "models/experiment_58_var_rat_random_3/", "models/experiment_58_var_rat_random_7/"]
path_bald = ["models/experiment_59_bald_random/", "models/experiment_59_bald_random_3/", "models/experiment_59_bald_random_7/"]
path_random = ["models/experiment_51_random_random/", "models/experiment_51_random_random_3/", "models/experiment_51_random_random_7/"]

final_results.append(statistical_significance(path_random, path_entropy, "random-simple", "entropy-simple", 27))
final_results.append(statistical_significance(path_random, path_var_rat, "random-simple", "variation-ratios-simple", 27))
final_results.append(statistical_significance(path_random, path_bald, "random-simple", "bald-simple", 27))


final_results.append("UKP")

path_entropy = ["models/experiment_67_entropy_random_10/", "models/experiment_67_entropy_random_13/", "models/experiment_67_entropy_random_14/"]
path_var_rat = ["models/experiment_68_var_rat_random_10/", "models/experiment_68_var_rat_random_13/", "models/experiment_68_var_rat_random_14/"]
path_bald = ["models/experiment_69_bald_random_10/", "models/experiment_69_bald_random_13/", "models/experiment_69_bald_random_14/"]
path_random = ["models/experiment_61_random_random_10/", "models/experiment_61_random_random_13/", "models/experiment_61_random_random_14/"]

final_results.append(statistical_significance(path_random, path_entropy, "random-simple", "entropy-simple", 27))
final_results.append(statistical_significance(path_random, path_var_rat, "random-simple", "variation-ratios-simple", 27))
final_results.append(statistical_significance(path_random, path_bald, "random-simple", "bald-simple", 27))


with open("results/significance_results_uncertainty_wilcoxon_with_variation.txt", "w") as file:
    for i in final_results:
        file.write(i+r"\\"+"\n"+r"\hline"+"\n")
