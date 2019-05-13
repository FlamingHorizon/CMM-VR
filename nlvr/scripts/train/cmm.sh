#!/bin/bash

checkpoint_path="data/film.pt"
log_path="data/film.log"
python scripts/train_model_raw.py \
  --checkpoint_path $checkpoint_path \
  --model_type FiLM \
  --num_iterations 20000000 \
  --print_verbose_every 20000000 \
  --checkpoint_every 1000 \
  --record_loss_every 100 \
  --num_val_samples 10000 \
  --num_test_samples 10000 \
  --optimizer Adam \
  --learning_rate 3.5e-4 \
  --batch_size 128 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 1 \
  --module_batchnorm 1 \
  --classifier_batchnorm 1 \
  --bidirectional 1 \
  --decoder_type linear \
  --encoder_type lstm \
  --weight_decay 1e-5 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 200 \
  --rnn_hidden_dim 256 \
  --rnn_output_batchnorm 0 \
  --classifier_downsample maxpoolfull \
  --classifier_proj_dim 512 \
  --classifier_fc_dims 1024 \
  --module_input_proj 1 \
  --module_residual 1 \
  --module_dim 128 \
  --module_dropout 0e-2 \
  --module_stem_kernel_size 3 \
  --module_kernel_size 3 \
  --module_batchnorm_affine 0 \
  --module_num_layers 1 \
  --num_modules 3 \
  --condition_pattern 1,1,1,1 \
  --gamma_option linear \
  --gamma_baseline 1 \
  --use_gamma 1 \
  --use_beta 1 \
  --condition_method bn-film \
  --program_generator_parameter_efficient 1 \
  --gpu_devices 0,1 \
  --gpu_visible 2,3 \
  --time 1\
  | tee $log_path
