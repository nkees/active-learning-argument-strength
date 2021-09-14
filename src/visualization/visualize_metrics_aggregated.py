import argparse
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from calculate_performance import (
    collect_single_model_performance,
    collect_multiple_model_performance,
    collect_subdirs,
)
import pandas as pd

# ------------------------------------------------------
# IBM, Uncertainty, test on all
# ------------------------------------------------------

plt.style.use("seaborn-whitegrid")

parser = argparse.ArgumentParser()
args = parser.parse_args()

path_entropy = ["models/experiment_57_entropy_random/", "models/experiment_57_entropy_random_3/", "models/experiment_57_entropy_random_7/"]
path_var_rat = ["models/experiment_58_var_rat_random/", "models/experiment_58_var_rat_random_3/", "models/experiment_58_var_rat_random_7/"]
path_bald = ["models/experiment_59_bald_random/", "models/experiment_59_bald_random_3/", "models/experiment_59_bald_random_7/"]
path_random = ["models/experiment_51_random_random/", "models/experiment_51_random_random_3/", "models/experiment_51_random_random_7/"]

df_path_list = [path_entropy, path_var_rat, path_bald, path_random]
df_list = []
for path in df_path_list:
    temp = pd.DataFrame(columns=["model_name", "num_test_data", "valid_acc", "eval_acc"])
    for path_path in path:
        temp = pd.concat([temp, collect_multiple_model_performance(path_path, args)])
    df_list.append(temp)

df_list_new = []

print("Paths initialized")
print("Collecting the metrics...")

for i in range(len(df_list)):
    print(f"    Collecting in {i}...")
    df = df_list[i][:]
    print(df)
    df_mean = df.groupby("num_test_data")["eval_acc"].mean()
    df_mean.index.name = "num_test_data"
    df_mean = df_mean.reset_index(drop=False)
    df_std = df.groupby("num_test_data")["eval_acc"].std()
    df_std.index.name = "num_test_data"
    df_std = df_std.reset_index(drop=False)

    df_mean["eval_acc_std"] = df_std["eval_acc"]

    alpha = 0.05
    size = 30

    df_mean["upper_mean_eval_acc"] = df_mean["eval_acc"] + stats.t.ppf(
        1.0 - (alpha / 2.0), size - 1
    ) * (df_mean["eval_acc_std"] / np.sqrt(size))
    df_mean["lower_mean_eval_acc"] = df_mean["eval_acc"] - stats.t.ppf(
        1.0 - (alpha / 2.0), size - 1
    ) * (df_mean["eval_acc_std"] / np.sqrt(size))
    df = df_mean[:]
    df_list_new.append(df)
    print(len(df), df.head())

print("Metrics collected. Building graphs.")
plt.figure(figsize=(15, 10), facecolor="white")
plt.xlim(100, 3640)
plt.ylim(0.5, 0.9)
plt.plot(
    df_list_new[0]["num_test_data"],
    df_list_new[0]["eval_acc"],
    marker="",
    color="orchid",
    linewidth=2,
    alpha=1,
    label="entropy",
)

plt.fill_between(
    df_list_new[0]["num_test_data"],
    df_list_new[0]["upper_mean_eval_acc"],
    df_list_new[0]["lower_mean_eval_acc"],
    color="orchid",
    alpha=0.1,
)

plt.plot(
    df_list_new[1]["num_test_data"],
    df_list_new[1]["eval_acc"],
    marker="",
    color="olive",
    linewidth=2,
    alpha=1,
    label="variation-ratios",
)

plt.fill_between(
    df_list_new[1]["num_test_data"],
    df_list_new[1]["upper_mean_eval_acc"],
    df_list_new[1]["lower_mean_eval_acc"],
    color="olive",
    alpha=0.1,
)

plt.plot(
    df_list_new[2]["num_test_data"],
    df_list_new[2]["eval_acc"],
    marker="",
    color="darkorange",
    linewidth=2,
    alpha=1,
    label="bald",
)

plt.fill_between(
    df_list_new[2]["num_test_data"],
    df_list_new[2]["upper_mean_eval_acc"],
    df_list_new[2]["lower_mean_eval_acc"],
    color="darkorange",
    alpha=0.1,
)

plt.plot(
    df_list_new[3]["num_test_data"],
    df_list_new[3]["eval_acc"],
    marker="",
    color="slateblue",
    linewidth=2,
    alpha=1,
    label="random",
)

plt.fill_between(
    df_list_new[3]["num_test_data"],
    df_list_new[3]["upper_mean_eval_acc"],
    df_list_new[3]["lower_mean_eval_acc"],
    color="slateblue",
    alpha=0.1,
)

plt.xticks(range(0, 3640, 500))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("\nTraining data size", fontsize=15)
plt.ylabel("Accuracy\n", fontsize=15)
plt.legend(loc="lower right", frameon=True, ncol=2, fontsize=15)
plt.suptitle(
    "Model performance with uncertainty acquisition - IBM, test on all (3, 4, 7), aggregated",
    fontsize=15,
    fontweight=0,
    color="grey",
)
plt.savefig("graphs/performance_uncertainty_combined_zoomed_in_IBM_t-all.png")
# plt.show()
print("Complete.")

# ------------------------------------------------------
# UKP, Uncertainty, test on all
# ------------------------------------------------------

plt.style.use("seaborn-whitegrid")

parser = argparse.ArgumentParser()
args = parser.parse_args()

path_entropy = ["models/experiment_67_entropy_random_10/", "models/experiment_67_entropy_random_13/", "models/experiment_67_entropy_random_14/"]
path_var_rat = ["models/experiment_68_var_rat_random_10/", "models/experiment_68_var_rat_random_13/", "models/experiment_68_var_rat_random_14/"]
path_bald = ["models/experiment_69_bald_random_10/", "models/experiment_69_bald_random_13/", "models/experiment_69_bald_random_14/"]
path_random = ["models/experiment_61_random_random_10/", "models/experiment_61_random_random_13/", "models/experiment_61_random_random_14/"]

df_path_list = [path_entropy, path_var_rat, path_bald, path_random]
df_list = []
for path in df_path_list:
    temp = pd.DataFrame(columns=["model_name", "num_test_data", "valid_acc", "eval_acc"])
    for path_path in path:
        temp = pd.concat([temp, collect_multiple_model_performance(path_path, args)])
    df_list.append(temp)

df_list_new = []

print("Paths initialized")
print("Collecting the metrics...")

for i in range(len(df_list)):
    print(f"    Collecting in {i}...")
    df = df_list[i][:]
    print(df)
    df_mean = df.groupby("num_test_data")["eval_acc"].mean()
    df_mean.index.name = "num_test_data"
    df_mean = df_mean.reset_index(drop=False)
    df_std = df.groupby("num_test_data")["eval_acc"].std()
    df_std.index.name = "num_test_data"
    df_std = df_std.reset_index(drop=False)

    df_mean["eval_acc_std"] = df_std["eval_acc"]

    alpha = 0.05
    size = 30

    df_mean["upper_mean_eval_acc"] = df_mean["eval_acc"] + stats.t.ppf(
        1.0 - (alpha / 2.0), size - 1
    ) * (df_mean["eval_acc_std"] / np.sqrt(size))
    df_mean["lower_mean_eval_acc"] = df_mean["eval_acc"] - stats.t.ppf(
        1.0 - (alpha / 2.0), size - 1
    ) * (df_mean["eval_acc_std"] / np.sqrt(size))
    df = df_mean[:]
    df_list_new.append(df)
    print(len(df), df.head())

print("Metrics collected. Building graphs.")
plt.figure(figsize=(15, 10), facecolor="white")
plt.xlim(100, 3640)
plt.ylim(0.5, 0.9)
plt.plot(
    df_list_new[0]["num_test_data"],
    df_list_new[0]["eval_acc"],
    marker="",
    color="orchid",
    linewidth=2,
    alpha=1,
    label="entropy",
)

plt.fill_between(
    df_list_new[0]["num_test_data"],
    df_list_new[0]["upper_mean_eval_acc"],
    df_list_new[0]["lower_mean_eval_acc"],
    color="orchid",
    alpha=0.1,
)

plt.plot(
    df_list_new[1]["num_test_data"],
    df_list_new[1]["eval_acc"],
    marker="",
    color="olive",
    linewidth=2,
    alpha=1,
    label="variation-ratios",
)

plt.fill_between(
    df_list_new[1]["num_test_data"],
    df_list_new[1]["upper_mean_eval_acc"],
    df_list_new[1]["lower_mean_eval_acc"],
    color="olive",
    alpha=0.1,
)

plt.plot(
    df_list_new[2]["num_test_data"],
    df_list_new[2]["eval_acc"],
    marker="",
    color="darkorange",
    linewidth=2,
    alpha=1,
    label="bald",
)

plt.fill_between(
    df_list_new[2]["num_test_data"],
    df_list_new[2]["upper_mean_eval_acc"],
    df_list_new[2]["lower_mean_eval_acc"],
    color="darkorange",
    alpha=0.1,
)

plt.plot(
    df_list_new[3]["num_test_data"],
    df_list_new[3]["eval_acc"],
    marker="",
    color="slateblue",
    linewidth=2,
    alpha=1,
    label="random",
)

plt.fill_between(
    df_list_new[3]["num_test_data"],
    df_list_new[3]["upper_mean_eval_acc"],
    df_list_new[3]["lower_mean_eval_acc"],
    color="slateblue",
    alpha=0.1,
)

plt.xticks(range(0, 3640, 500))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("\nTraining data size", fontsize=15)
plt.ylabel("Accuracy\n", fontsize=15)
plt.legend(loc="lower right", frameon=True, ncol=2, fontsize=15)
plt.suptitle(
    "Model performance with uncertainty acquisition - UKP, test on all (10, 13, 14), aggregated",
    fontsize=15,
    fontweight=0,
    color="grey",
)
plt.savefig("graphs/performance_uncertainty_combined_zoomed_in_UKP_t-all.png")
# plt.show()
print("Complete.")
