import os
import pandas as pd
import logging 


def check_repetitions(data_set):
    id_list = []
    repetitions = 0
    for i in range(len(data_set)):
        args = data_set.iloc[i].id
        if args in id_list:
            # print(f"REPETITION! {args}")
            repetitions += 1
        else: 
            id_list.append(args)
    print(f"{repetitions} repetitions have been found.")
    return repetitions, len(id_list)


def check_integrity_of_data_pool(path):
    # path = os.path.join("models", "experiment_1_random_graph", "model1", "active_data")
    all_data = pd.read_csv(os.path.join(path, "training_pool_all.csv"))
    training_data = pd.read_csv(os.path.join(path, "training_subset.csv"))
    remaining_data = pd.read_csv(os.path.join(path, "remaining_training_pool.csv"))
    filtered_data = pd.read_csv(os.path.join(path, "filtered_out.csv"))
    logging.info("All data: " + str(len(all_data)))
    logging.info("Used training data: " + str(len(training_data)))
    logging.info("Remaining data: " + str(len(remaining_data)))
    logging.info("Filtered data: " + str(len(filtered_data)))
    logging.info("All data minus training and remaining data: " + str(len(all_data) - len(training_data) - len(remaining_data)))
    repetitions, should_be_filtered = check_repetitions(filtered_data)
    logging.info("Unique pairs in filtered_data: " + str(len(filtered_data) - repetitions))
    logging.info("The core of the filtered data: " + str(should_be_filtered))
    #assert (len(training_data) + len(remaining_data) + len(filtered_data) - repetitions) == len(all_data)

if __name__ == "__main__":
    check_integrity_of_data_pool()
    #print("all is ok")