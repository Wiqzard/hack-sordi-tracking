export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --is_training 1 \
  --model_id faster_rcnn_train1 \
  --model Faster_RCNN\
  --data custom \
  --root_path ./data/ \
  --ignore_redundant \
  --partion_single_assets 2 \
  --ratio 0.8 \
  --checkpoints ./checkpoints/ \
  --batch_size 2 \
  --num_workers 2 \
  --itr 1 \
  --train_epochs 500 \
  --patience 7 \
  --optimizer "sgd" \
  --momentum 0.9 \
  --weight_decay 0.0005 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --loss 'mse' \
  --lradj 'type1' \
  --use_gpu True \
  --use_amp  \
  --gpu 1 \
  
  --devices 1
  --use_multi_gpu False \
 


python -u run.py \
  --is_training 1 \
  --model_id faster_rcnn_train1 \
  --model Faster_RCNN\
  --data custom \
  --root_path ./data/ \
  --ignore_redundant \
  --partion_single_assets 2 \
  --ratio 0.8 \
  --checkpoints ./checkpoints/ \
  --batch_size 3 \
  --num_workers 0 \
  --itr 1 \
  --train_epochs 2 \
  --patience 7 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --loss 'mse' \
  --lradj 'type1' \
  --use_gpu True \
  --gpu 0 \
  --devices 0



  --use_multi_gpu False \
  --use_amp False \