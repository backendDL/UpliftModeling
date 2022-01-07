# python train.py --seed 42 --do_train --do_eval --do_predict \
#  --data_path ./dataset/dataset_6_1_168 --use_weighted_sampler \
#  --num_epochs 50 --learning_rate 1e-3 --weight_decay 0.01 \
#  --propensity_score 0.5 --alpha 0.7 \
#  --per_device_train_batch_size 256 --per_device_eval_batch_size 1024 \
#  --model_type tcn --tcn_num_channels 32 32 32 32 32 32 32 32 32 32 \
#  --tcn_out_features 32 --tcn_dropout 0.2 --cutoff 0.2 \
#  --tcn_num_layers 1 --tcn_num_ensembles 1 \
#  --max_saved_models 5 --best_metric qini --higher_the_better

python train.py --seed 42 --do_train --do_eval \
 --data_path ./dataset2/dataset_6_3_168 --use_weighted_sampler \
 --num_epochs 10 --learning_rate 1e-3 --weight_decay 0.01 \
 --propensity_score 0.5 --alpha 0.7 \
 --per_device_train_batch_size 256 --per_device_eval_batch_size 256 \
 --model_type tcn --tcn_num_channels 16 16 16 16 16 16 16 16 16 16 \
 --tcn_out_features 16 --tcn_dropout 0.2 --cutoff 0.2 \
 --tcn_num_layers 1 --tcn_num_ensembles 1 \
 --max_saved_models 5 --best_metric qini --higher_the_better