python train.py --seed 42 --data_path ./dataset/dataset_6_1_168 --use_weighted_sampler --do_train --do_eval \
 --num_epochs 10 --learning_rate 1e-3 --weight_decay 0.01 --propensity_score 0.5 --alpha 0.1 \
 --per_device_train_batch_size 256 --per_device_eval_batch_size 1024 \
 --model_type tcn --rnn_out_features 16 --tcn_num_channels 16 16 16 16 16 16 16 16 16 16 \
 --tcn_out_features 16 --tcn_dropout 0.2 --cutoff 0.2 --max_saved_models 5 --best_metric auuc --higher_the_better

python train.py --seed 42 --data_path ./dataset/dataset_6_1_168 --use_weighted_sampler --do_train --do_eval \
 --num_epochs 10 --learning_rate 1e-3 --weight_decay 0.01 --propensity_score 0.5 --alpha 0.3 \
 --per_device_train_batch_size 256 --per_device_eval_batch_size 1024 \
 --model_type tcn --rnn_out_features 16 --tcn_num_channels 16 16 16 16 16 16 16 16 16 16 \
 --tcn_out_features 16 --tcn_dropout 0.2 --cutoff 0.2 --max_saved_models 5 --best_metric auuc --higher_the_better

python train.py --seed 42 --data_path ./dataset/dataset_6_1_168 --use_weighted_sampler --do_train --do_eval \
 --num_epochs 10 --learning_rate 1e-3 --weight_decay 0.01 --propensity_score 0.5 --alpha 0.5 \
 --per_device_train_batch_size 256 --per_device_eval_batch_size 1024 \
 --model_type tcn --rnn_out_features 16 --tcn_num_channels 16 16 16 16 16 16 16 16 16 16 \
 --tcn_out_features 16 --tcn_dropout 0.2 --cutoff 0.2 --max_saved_models 5 --best_metric auuc --higher_the_better

python train.py --seed 42 --data_path ./dataset/dataset_6_1_168 --use_weighted_sampler --do_train --do_eval \
 --num_epochs 10 --learning_rate 1e-3 --weight_decay 0.01 --propensity_score 0.5 --alpha 0.7 \
 --per_device_train_batch_size 256 --per_device_eval_batch_size 1024 \
 --model_type tcn --rnn_out_features 16 --tcn_num_channels 16 16 16 16 16 16 16 16 16 16 \
 --tcn_out_features 16 --tcn_dropout 0.2 --cutoff 0.2 --max_saved_models 5 --best_metric auuc --higher_the_better

python train.py --seed 42 --data_path ./dataset/dataset_6_1_168 --use_weighted_sampler --do_train --do_eval \
 --num_epochs 10 --learning_rate 1e-3 --weight_decay 0.01 --propensity_score 0.5 --alpha 0.9 \
 --per_device_train_batch_size 256 --per_device_eval_batch_size 1024 \
 --model_type tcn --rnn_out_features 16 --tcn_num_channels 16 16 16 16 16 16 16 16 16 16 \
 --tcn_out_features 16 --tcn_dropout 0.2 --cutoff 0.2 --max_saved_models 5 --best_metric auuc --higher_the_better