import os
from src.finetuning.file_management import FileManager
from src.finetuning.initializer import initialize_with_mask

acquisition_functions = ["entropy", "bald", "variation_ratios"]


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
