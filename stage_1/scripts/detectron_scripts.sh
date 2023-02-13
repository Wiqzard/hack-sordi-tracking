
python -u main.py \
    --inference True \
    --is_training True \
    --test False \
    --register_data True \
    --test False \
    --model_checkpoint "" \
    --root_path ./data/ \
    --ignore_redundant \
    --partion_single_assets 3\
    --ratio 0.98 \
    --area_threshold_min 6000 \
    --area_threshold_max 700000 \
    --model "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" \
    --num_workers 4 \
    --ims_per_batch 2 \
    --base_lr 0.001 \
    --warmup_steps 1000\
    --max_iter 100000\
    --batch_per_img 2048 \
    --use_gpu \
    --checkpoint_period 2000 \
    --use_amp
    --resume

    --eval_period 5000 \
  #--resume
    --use_amp \


    --nms_threshold 0.1 \
    --gamma 0.1 \
    --writer_period 100 \
    --patience 1000 \


   