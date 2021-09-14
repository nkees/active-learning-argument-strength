# Active learning pipeline
Code to the paper "Active Learning for Argument Strength Estimation".

The following instructions allow to replicate the procedures described in the paper.

## Set up
Use conda as your environment manager:
```
conda env create --file environment.yml
```
Activate your environment, e.g.:
```
conda activate env
```

## Training
Run the active learning pipeline with the following arguments:

### IBM
Source:
https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml

#### ACTIVE LEARNING, Uncertainty-based

Note: use `--num_train_epochs=9999.0` for early stopping. Otherwise, stipulate your own value in this field. 

##### TEST ON TOPIC 4

```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_51_random_4/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_57_entropy_4/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_58_var_rat_4/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_59_bald_4/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```

##### TEST ON TOPIC 3
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_51_random_3/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_57_entropy_3/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_58_var_rat_3/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_59_bald_3/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```

##### TEST ON TOPIC 7
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_51_random_7/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_57_entropy_7/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_58_var_rat_7/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_59_bald_7/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```

#### PASSIVE LEARNING (ON THE WHOLE DATA)
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ibm_pairs.npy --output_dir=models/experiment_00_4/ --num_train_epochs=9999.0 --sample_models=10
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_00_3/ --num_train_epochs=9999.0 --sample_models=10
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_00_7/ --num_train_epochs=9999.0 --sample_models=10
```


### UKP

Source: https://github.com/UKPLab/acl2016-convincing-arguments/tree/master/data 

#### ACTIVE LEARNING, Uncertainty-based
##### TEST ON TOPIC 13
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_61_random_13/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_67_entropy_13/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_68_var_rat_13/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_69_bald_13/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```

##### TEST ON TOPIC 14
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_61_random_14/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_67_entropy_14/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_68_var_rat_14/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_69_bald_14/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```

##### TEST ON TOPIC 10
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_61_random_10/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_67_entropy_10/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_68_var_rat_10/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_69_bald_10/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```


#### PASSIVE LEARNING (ON THE WHOLE DATA)
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_10_13/ --num_train_epochs=9999.0 --sample_models=10
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_10_14/ --num_train_epochs=9999.0 --sample_models=10
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_10_10/ --num_train_epochs=9999.0 --sample_models=10
```

## Evaluation
e.g.
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=models/experiment305/model10/checkpoint-275 --task_name=AP --learning_rate=2e-5 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_eval --output_dir=models/experiment_test01/ 
```

## Statistical Analysis and Visualization
For running the analysis, you have to store the models and their supporting files in the `models/` directory.
You can access our experiment results, though, in the folders `results/` and `graphs/`.

To perform the statistical analysis of the results, run the following line, having previously adjusted the paths in the respective file:

```
python src/visualization/significance_uncertainty_based.py
```
The results will be stored in the folder `results/`.

Visualization of the experiment results is possible by running this line:

```
python src/visualization/visualize_metrics_aggregated.py
```
The results will be stored in the folder `graphs/`.

## Unit testing
```
python -m unittest discover -s src/testing
```

## Delete models (clear up space)
```
python src/postprocessing/clean_up.py --models_dir=models/experiment_9_bald_random
```


## Other
Trained on a single NVIDIA RTX 2080 GPU. 
