import os
import numpy as np
import logging
from src.finetuning.graph_structure import Graph, retrieve_argument_ids

logger = logging.getLogger(__name__)


def use_mask(mask_path):
    return np.load(mask_path)


def save_mask(path_to, mask):
    return np.save(path_to, mask)


def initialize_with_mask(df, num_data_points, path, name):
    """
    Initializes a mask for train-validation-test split of a given set of data
    or if already initialized, uses a mask which is already there for a given set of data
    Args:
        df: data frame with the data to split
        num_data_points: size of the initial data set
        path: where should the mask be stored/found
        name: how should the mask be called

    Returns: a data frame of size num_data_points as an initial data frame

    """
    name = f"{name}.npy"
    mask_path = os.path.join(path, name)
    logger.info(f"Looking for mask at {mask_path}")
    if os.path.exists(mask_path):
        mask = use_mask(mask_path)
    else:
        mask = create_and_save_initialization_mask(df, num_data_points, path, name)
    df_subset = df[mask]
    assert len(df_subset) == num_data_points
    return df_subset


def create_and_save_initialization_mask(df, num_data_points, path_to, name):
    """
    Initialize a sample and save its mask. For higher diversity of the initial dataset,
    graph_method is used: if a candidate sample contains redundant argument pairs, the label
    of which can be inferred from other argument pairs, the sample will be drawn anew.
    """
    df_subset = df.sample(n=num_data_points)
    while contains_connected_arguments(df_subset):
        df_subset = df.sample(n=num_data_points)  # sample a new one
    id_list = df_subset["id"].to_list()
    mask = np.array(df["id"].isin(id_list))
    name = f"{name}"
    if not os.path.exists(path_to):
        os.makedirs(path_to)
    path_to = os.path.join(path_to, name)
    save_mask(path_to, mask)
    return mask


def contains_connected_arguments(df):
    graph_temp = Graph()
    for index in range(len(df)):
        df_one_element = df.iloc[[index]]
        argument1, argument2 = retrieve_argument_ids(df_one_element)
        if graph_temp.is_in_graph(argument1, argument2):
            return True
        else:
            graph_temp.push_argument_pair_to_graph(argument1, argument2)
    return False
