# Active learning pipeline
Code to the thesis "Active Learning for Argument Strength Prediction",
written at the Ludwig-Maximilians-Universität München, Institut für Informatik.

- Author: Nataliia Kees

- Department Head: PROF. DR. THOMAS SEIDL

- Supervisor: MICHAEL FROMM

- Submission date: 04.09.2020

Abstract:

Active learning as a means of dramatically reducing the amount of required
labelled data by applying smart data selection techniques is a vivid research
area, but completely under-researched in the eld of argument strength prediction.
This thesis investigates acquisition functions for batch sampling in
active learning for pairwise relative argument strength prediction task. For
this, we test heuristics which are expected to guide a learner to select the most
informative data and reach best performance with the least labelled data. Our
model architecture is based on BERT Base Uncased with dropout at test time,
which makes it possible to approximate probabilistic processes and test uncertainty
sampling methods for active learning, and is applied on two data sets we
use: UKPConvArg1Strict and IBM-9.1kPairs. In addition to this, we develop
and investigate some exploration acquisition strategies, which are diversity and
density-based. We find that when compared to random data acquisition, none
of the heuristics perform better, rendering random acquisition the best strategy
among the strategies tested for pairwise argument strength prediction.


The following instructions allow to replicate the procedures conducted in the thesis.

## Set up
Use conda as your environment manager:
```
conda env create --file environment.yml
```
Activate your environment, e.g.:
```
conda activate /mnt/data1/shvets/anaconda3/envs/argenv
```
When changing anything in the virtual environment, export it to the `environment.yml`:
```
conda env export --no-builds > environment.yml
```

## Training
Run the active learning pipeline with the following arguments (for AP task and testing on one middle-sized topic):

### IBM
Source:
https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml

#### ACTIVE LEARNING

##### TEST ON TOPIC 4
###### Simple
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_51_random_random/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_57_entropy_random/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_58_var_rat_random/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_59_bald_random/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```
###### Graph methods
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_51_random_graph/ --acq_func=random --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_57_entropy_graph/ --acq_func=entropy --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_58_var_rat_graph/ --acq_func=variation_ratios --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_59_bald_graph/ --acq_func=bald --sample_models=10 --graph_method --stopping_criterion=27
```
###### Graph scoring
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_53_scoring_4/ --sample_models=10 --graph_scoring --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_53_scoring_transitive_4/ --sample_models=10 --graph_scoring --transitivity_method --stopping_criterion=27
```
##### For stopping when reached specific performance:
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_4.npy --output_dir=models/experiment_59_bald_random/ --acq_func=bald --sample_models=10 --stopping_criterion="performance reached" --passive_path="models/experiment_00/"
```

##### TEST ON TOPIC 3
###### Simple
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_51_random_random_3/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_57_entropy_random_3/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_58_var_rat_random_3/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_59_bald_random_3/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```
###### Graph methods
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_51_random_graph_3/ --acq_func=random --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_57_entropy_graph_3/ --acq_func=entropy --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_58_var_rat_graph_3/ --acq_func=variation_ratios --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_59_bald_graph_3/ --acq_func=bald --sample_models=10 --graph_method --stopping_criterion=27
```
###### Graph scoring
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_53_scoring_3/ --sample_models=10 --graph_scoring --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_53_scoring_transitive_3/ --sample_models=10 --graph_scoring --transitivity_method --stopping_criterion=27
```

##### TEST ON TOPIC 7
###### Simple
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_51_random_random_7/ --acq_func=random --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_57_entropy_random_7/ --acq_func=entropy --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_58_var_rat_random_7/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_59_bald_random_7/ --acq_func=bald --sample_models=10 --stopping_criterion=27
```
###### Graph methods
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_51_random_graph_7/ --acq_func=random --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_57_entropy_graph_7/ --acq_func=entropy --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_58_var_rat_graph_7/ --acq_func=variation_ratios --sample_models=10 --graph_method --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_59_bald_graph_7/ --acq_func=bald --sample_models=10 --graph_method --stopping_criterion=27
```
###### Graph scoring
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_53_scoring_7/ --sample_models=10 --graph_scoring --stopping_criterion=27
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_53_scoring_transitive_7/ --sample_models=10 --graph_scoring --transitivity_method --stopping_criterion=27
```

#### PASSIVE LEARNING
```
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=4 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ibm_pairs.npy --output_dir=models/experiment_00/ --num_train_epochs=9999.0 --sample_models=10
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=3 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ibm_pairs_3.npy --output_dir=models/experiment_00_3/ --num_train_epochs=9999.0 --sample_models=10
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=7 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ibm_pairs_7.npy --output_dir=models/experiment_00_7/ --num_train_epochs=9999.0 --sample_models=10
python trainable.py --data_dir=data/IBM-9.1kPairs/complete_new_split/complete.csv --test=-1 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --logging_steps=-2 --mask_path=masks/split_ibm_pairs_-1.npy --output_dir=models/experiment_000_t-all/ --num_train_epochs=9999.0 --sample_models=10
```


### UKP
Source: https://github.com/UKPLab/acl2016-convincing-arguments/tree/master/data 
#### ACTIVE LEARNING
##### TEST ON TOPIC 13
###### Simple
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_61_random_random_13/ --acq_func=random --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_67_entropy_random_13/ --acq_func=entropy --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_68_var_rat_random_13/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_69_bald_random_13/ --acq_func=bald --sample_models=10 --stopping_criterion=35
```
###### Graph methods
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_61_random_graph_13/ --acq_func=random --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_67_entropy_graph_13/ --acq_func=entropy --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_68_var_rat_graph_13/ --acq_func=variation_ratios --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_69_bald_graph_13/ --acq_func=bald --sample_models=10 --graph_method --stopping_criterion=35
```
###### Graph scoring
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_63_scoring_13/ --sample_models=10 --graph_scoring --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=13 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_13.npy --output_dir=models/experiment_63_scoring_transitive_13/ --sample_models=10 --graph_scoring --transitivity_method --stopping_criterion=27
```

##### TEST ON TOPIC 14
###### Simple
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_61_random_random_14/ --acq_func=random --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_67_entropy_random_14/ --acq_func=entropy --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_68_var_rat_random_14/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_69_bald_random_14/ --acq_func=bald --sample_models=10 --stopping_criterion=35
```
###### Graph method
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_61_random_graph_14/ --acq_func=random --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_67_entropy_graph_14/ --acq_func=entropy --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_68_var_rat_graph_14/ --acq_func=variation_ratios --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_69_bald_graph_14/ --acq_func=bald --sample_models=10 --graph_method --stopping_criterion=35
```
###### Graph scoring
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_63_scoring_14/ --sample_models=10 --graph_scoring --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=14 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_14.npy --output_dir=models/experiment_63_scoring_transitive_14/ --sample_models=10 --graph_scoring --transitivity_method --stopping_criterion=27
```
##### TEST ON TOPIC 10
###### Simple
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_61_random_random_10/ --acq_func=random --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_67_entropy_random_10/ --acq_func=entropy --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_68_var_rat_random_10/ --acq_func=variation_ratios --sample_models=10 --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_69_bald_random_10/ --acq_func=bald --sample_models=10 --stopping_criterion=35
```
###### Graph method
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_61_random_graph_10/ --acq_func=random --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_67_entropy_graph_10/ --acq_func=entropy --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_68_var_rat_graph_10/ --acq_func=variation_ratios --sample_models=10 --graph_method --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --inference_iterations=20 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_69_bald_graph_10/ --acq_func=bald --sample_models=10 --graph_method --stopping_criterion=35
```
###### Graph scoring
```
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_63_scoring_10/ --sample_models=10 --graph_scoring --stopping_criterion=35
python trainable.py --data_dir=data/UKPConvArg1Strict-CSV/complete_new_split/complete.csv --test=10 --validation_size=0.15 --model_type=bert --model_name=bert-base-uncased --task_name=AP --learning_rate=2e-5 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=200 --max_seq_length=80 --do_train --do_eval --active_learning --logging_steps=-2 --active_learning_batch_size=130 --num_train_epochs=9999.0 --mask_path=masks/split_ukp_pairs_10.npy --output_dir=models/experiment_63_scoring_transitive_10/ --sample_models=10 --graph_scoring --transitivity_method --stopping_criterion=27
```

#### PASSIVE LEARNING
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

## Visualization

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
