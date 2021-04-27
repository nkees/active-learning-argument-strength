import pandas as pd
import logging
import os
import tqdm
from src.finetuning.graph_structure import Graph, retrieve_argument_ids
from src.finetuning.file_management import FileManager
from src.finetuning.initializer import initialize_with_mask
from src.finetuning.graph_scorer import calculate_scores, calculate_candidate_score
import multiprocessing as mp
from joblib import Parallel, delayed

num_cores = mp.cpu_count()

acquisition_functions = ["entropy", "bald", "variation_ratios"]


def process(i, data_pool, N, graphs, elements_with_scores_all, pbar):
    row = data_pool.loc[[i]]
    arg1, arg2 = retrieve_argument_ids(row)
    topic = row["new_topic_id"].iloc[0]
    result = calculate_candidate_score(N, (arg1, arg2), graphs[topic], elements_with_scores_all[topic])
    pbar.update(1)
    id = f"{arg1}_{arg2}"
    return result, id


class AcquisitionClass:
    """
    Class for pipeline initialization and inheritance by the other pipelines.
    """

    def __init__(self, acquisition_function, data_path):
        self.acquisition_function = acquisition_function
        self.data_path = data_path
        self.training_pool_all = FileManager(self.data_path, "training_pool_all.csv")
        self.training_subset = FileManager(self.data_path, "training_subset.csv")
        self.remaining_training_pool = FileManager(self.data_path, "remaining_training_pool.csv")

    def initialize_data_pool(self, num_data_points, model_n, test_n, dataset_name):
        data_pool = self.training_pool_all.read()
        self.remaining_training_pool.save_with_replacement(data_pool)
        masks_path = os.path.join("masks", "initialization_masks", dataset_name)
        name = f"init_mask_{model_n}_test_on_{test_n}"
        data_subset = initialize_with_mask(data_pool, num_data_points, masks_path, name)
        data_subset = data_subset.reset_index(drop=True)
        return data_subset

    @staticmethod
    def get_samples_random(data_pool, num_data_points):
        df = data_pool.sample(n=num_data_points)
        return df

########################################################################################################################


class RandomAcquisition(AcquisitionClass):
    """
    This is the baseline which is compared against all the
    other measures (random-simple). The expectation is that other
    measures will outperform the random acquisition and our goal is to test whether
    this is the case and if yes, for which heuristics in particular.
    """

    def __init__(self, acquisition_function, output_data_dir):
        AcquisitionClass.__init__(self, acquisition_function, output_data_dir)

    def get_data_subset(self, num_data_points, model_n, test_n, dataset_name):
        if self.remaining_training_pool.exists():
            data_pool = self.remaining_training_pool.read()
            if len(data_pool) < num_data_points:
                df = data_pool
            else:
                df = super().get_samples_random(data_pool, num_data_points)
        else:
            df = super().initialize_data_pool(num_data_points, model_n, test_n, dataset_name)
            data_pool = self.remaining_training_pool.read()

        df = self.training_subset.append_and_save(df)
        used_ids = self.training_subset.get_ids()
        remaining_data = data_pool[~(data_pool["id"].isin(used_ids))]
        self.remaining_training_pool.save_with_replacement(remaining_data)
        return df

########################################################################################################################


class UncertaintyAcquisition(AcquisitionClass):
    """
    Data acquisition with uncertainty-based methods.
    Possible heuristics:
        predictive entropy (entropy-simple);
        variation ratios (variation-ratios-simple);
        mutual information (BALD) (bald-simple).

    """

    def __init__(self, acquisition_function, output_data_dir):
        AcquisitionClass.__init__(self, acquisition_function, output_data_dir)
        self.uncertain_data = FileManager(output_data_dir, "uncertain_data.csv")

        if self.acquisition_function not in acquisition_functions:
            raise NotImplementedError(f"This acquisition function has not been implemented yet. Please select one from the list: {acquisition_functions}")

    def get_data_subset(self, num_data_points, model_n, test_n, dataset_name):
        if self.remaining_training_pool.exists():
            data_pool = self.remaining_training_pool.read()
            if len(data_pool) < num_data_points:
                df = data_pool
            else:
                df = self.get_samples_uncertainty_based(num_data_points)
        else:
            df = super().initialize_data_pool(num_data_points, model_n, test_n, dataset_name)
            data_pool = self.remaining_training_pool.read()

        df = self.training_subset.append_and_save(df)
        used_ids = self.training_subset.get_ids()
        remaining_data = data_pool[~(data_pool["id"].isin(used_ids))]
        self.remaining_training_pool.save_with_replacement(remaining_data)
        return df

    def get_samples_uncertainty_based(self, num_data_points):
        data_pool = self.uncertain_data.read()
        df = data_pool.iloc[0:num_data_points]
        return df

########################################################################################################################


class GraphDiversityAcquisition(AcquisitionClass):

    """
    This is our newly developed heuristic based
    on the characteristic of transitivity which our data points possess (random-
    graph). Transitivity allows us to model relations between single arguments
    with regards to their strength and make use of these relations as a proxy for
    diversity. The goal in each acquisition round is to select a set of unlabelled
    argument pairs, where the relation of argument 1 to argument 2 (and vice
    versa) cannot be derived from transitive relations already found in the training
    data set from the previous learning round, as well as in the newly acquired
    sample itself.
     We call it graph-based acquisition, because the arguments which have
    been already trained on are organized as a graph consisting of single arguments
    with transitive relations between them, where higher standing elements are
    stronger than the lower standing elements.
    Has an option of random selection (random-graph), or
    graph based acquisition with uncertrainty
    (entropy-graph, variation-ratios-graph, or bald-graph).

    """

    def __init__(self, acquisition_function, output_data_dir, ordered_graph):
        AcquisitionClass.__init__(self, acquisition_function, output_data_dir)
        self.filtered_out = FileManager(output_data_dir, "filtered_out.csv")
        self.graph = ordered_graph
        if self.acquisition_function == "random":
            self.random = True
        elif self.acquisition_function in acquisition_functions:
            self.random = False
            self.uncertain_data = FileManager(output_data_dir, "uncertain_data.csv")
        else:
            raise NotImplementedError(
                f"This acquisition function has not been implemented yet. Please select one from the list: {acquisition_functions} or do 'random'.")

    def initialize_filtered_df(self):
        df_filtered_out = pd.DataFrame(
            columns=["id", "a1", "a2", "label", "topic", "new_topic_id", self.acquisition_function])
        self.filtered_out.save_with_replacement(df_filtered_out)
        return df_filtered_out

    def get_data_subset(self, num_data_points, model_n, test_n, dataset_name):
        if self.remaining_training_pool.exists():
            data_pool = self.remaining_training_pool.read()

            if self.random or (not self.random and not self.uncertain_data.exists()):
                df, graph_temp = self.select_data_with_graph(num_data_points, data_pool, how="random")
            else:
                df, graph_temp = self.select_data_with_graph(num_data_points, data_pool, how="uncertain")

        else:
            graph_temp = Graph()
            df = super().initialize_data_pool(num_data_points, model_n, test_n, dataset_name)
            graph_temp.push_argument_pairs_from_df_to_graph(df)
            data_pool = self.remaining_training_pool.read()
            if not self.filtered_out.exists():
                self.initialize_filtered_df()

        self.graph.add_ordered_edges(graph_temp, df)  # after labeling by the oracle

        df = self.training_subset.append_and_save(df)
        used_ids_training_set = self.training_subset.get_ids()
        used_ids_filtered = self.filtered_out.get_ids()
        remaining_data = data_pool[~(data_pool["id"].isin(used_ids_training_set))]
        remaining_data = remaining_data[~(remaining_data["id"].isin(used_ids_filtered))]
        self.remaining_training_pool.save_with_replacement(remaining_data)
        return df, self.graph

    def select_data_with_graph(self, num_data_points, data_pool, how="random"):
        graph_temp = Graph()
        logging.info(f"Remaining training pool's length is {len(data_pool)} data points.")
        counter = num_data_points
        logging.info(f"{counter} data points will be acquired.")
        df = pd.DataFrame(columns=["id", "a1", "a2", "label", "topic", "new_topic_id", self.acquisition_function])

        if self.filtered_out.exists():
            df_filtered_out = self.filtered_out.read()
        else:
            df_filtered_out = self.initialize_filtered_df()

        if how == "uncertain":
            df_uncertain = self.uncertain_data.read()

        iteration = 1
        while counter != 0:
            logging.info(
                f"---------------------------- This is iteration number {iteration}.-----------------------------")
            if len(data_pool) == 0:
                logging.info(f"THERE ARE NO ELEMENTS LEFT; THE PROCESS WILL BE STOPPED")
                break
            if how == "random":
                df_one_element = data_pool.sample(n=1)
            elif how == "uncertain":  # if how == "uncertain"
                df_one_element = data_pool[data_pool["id"] == df_uncertain.iloc[0].id]
                df_one_element[self.acquisition_function] = df_uncertain.iloc[0][self.acquisition_function]
                df_uncertain = df_uncertain[1:].reset_index(drop=True)

            logging.info(f"{df_one_element} will be checked")
            argument1, argument2 = retrieve_argument_ids(df_one_element)
            id_ = str(argument1) + "_" + str(argument2)
            logging.info(f"Arguments {id_} have been picked.")
            data_pool = data_pool[~(data_pool.id == id_)].reset_index(drop=True)
            logging.info(f"Remaining training pool's length is now {len(data_pool)} data points.")

            if self.graph.is_in_graph(argument1, argument2) or graph_temp.is_in_graph(argument1, argument2):
                logging.info(f"Arguments {argument1, argument2} will be filtered out.")
                df_filtered_out = pd.concat([df_filtered_out, df_one_element])
                logging.info(f"{len(df_filtered_out)} have been filtered out as of now.")
            else:
                logging.info(f"Arguments: {argument1, argument2} will be added to the training set")
                graph_temp.push_argument_pair_to_graph(argument1, argument2)
                logging.info(f"Training pool's length was {len(df)} data points.")
                df = pd.concat([df, df_one_element])
                logging.info(f"    Now it has {len(df)} data points.")
                counter -= 1
                logging.info(f"One data point selected. {counter} more to go.")
            iteration += 1

        self.filtered_out.save_with_replacement(df_filtered_out)
        return df, graph_temp

########################################################################################################################


class GraphScoreAcquisition(AcquisitionClass):

    """
    Graph-based Scoring samples a given number of argument pair examples
    based on the estimated amount of transitive connections which can be gained
    if the given argument pair were selected for labelling (graph-scoring).
    This approach is dierent from the random-graph method, as instead of
    trying to collect most diverse examples, it focuses on building dense clusters
    of connected arguments.
    """

    def __init__(self, acquisition_function, output_data_dir):
        AcquisitionClass.__init__(self, acquisition_function, output_data_dir)

    def get_data_subset(self, num_data_points, model_n, test_n, dataset_name):
        if self.remaining_training_pool.exists():
            data_pool = self.remaining_training_pool.read()
            if "UKP" in dataset_name:
                max_num_topics = 16 - 1
            elif "IBM" in dataset_name:
                max_num_topics = 11 - 1
            df_training_subset = self.training_subset.read()
            if len(df_training_subset["new_topic_id"].unique()) < max_num_topics:
                df = super().get_samples_random(data_pool, num_data_points)
            else:
                df = self.select_data_with_graph(num_data_points, data_pool)
        else:
            df = super().initialize_data_pool(num_data_points, model_n, test_n, dataset_name)
            data_pool = self.remaining_training_pool.read()

        df = self.training_subset.append_and_save(df)
        used_ids_training_set = self.training_subset.get_ids()
        remaining_data = data_pool[~(data_pool["id"].isin(used_ids_training_set))]
        self.remaining_training_pool.save_with_replacement(remaining_data)
        return df

    def select_data_with_graph(self, num_data_points, data_pool):
        df_training_subset = self.training_subset.read()
        graph_remaining_data = Graph()
        graph_remaining_data.push_argument_pairs_from_df_to_graph(data_pool)
        graph_all = Graph(graph_remaining_data.view_as_tuples())
        graph_all.push_argument_pairs_from_df_to_graph(df_training_subset)
        N = graph_all.n

        graphs = {}
        elements_with_scores_all = {}
        for topic in df_training_subset["new_topic_id"].unique():
            df_topic = df_training_subset[df_training_subset["new_topic_id"] == topic].reset_index(drop=True)
            graph_topic = Graph()
            graph_topic.push_argument_pairs_from_df_to_graph(df_topic)
            graph_topic_transitive = graph_topic.transitive_closure()
            elements_with_scores = calculate_scores(graph_topic, N)
            graphs.update({topic: graph_topic_transitive})
            elements_with_scores_all.update({topic: elements_with_scores})

        inputs = tqdm.tqdm(range(len(data_pool)))

        scores, ids = zip(*Parallel(n_jobs=num_cores)(delayed(process)(i, data_pool, N, graphs, elements_with_scores_all, inputs) for i in inputs))

        candidate_list = list(zip(scores, ids))
        candidate_list_sorted = sorted(candidate_list, key=lambda x: x[0], reverse=True)
        selected_candidates = [candidate[1] for candidate in candidate_list_sorted[0:num_data_points]]
        df = data_pool[data_pool["id"].isin(selected_candidates)].reset_index(drop=True)
        return df