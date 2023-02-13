python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/dataset_1.yaml --img 1280 720 --cfg cfg/training/yolov7-e6e.yaml --weights ''--name yolov7-e6e-model_1 --hyp data/hyp.scratch.p6.yaml --epochs 150 --cache-images

python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/dataset_2.yaml  --img 1280 720 --cfg cfg/training/yolov7-e6e.yaml --weights ''--name yolov7-e6e-model_2 --hyp data/hyp.scratch.p6.yaml --epochs 150 --cache-images

python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/dataset_3.yaml  --img 1280 720 --cfg cfg/training/yolov7-e6e.yaml --weights ''--name yolov7-e6e-model_2 --hyp data/hyp.scratch.p6.yaml --epochs 150 --cache-images




python train.py --workers 8 --device 0 --batch-size 32 --data data/dataset_1.yaml --img 1280 720 --cfg cfg/training/yolov7x.yaml --weights ''--name yolov7-X-model_1 --hyp data/hyp.scratch.p5.yaml --epochs 150 --cache-images

python train.py --workers 8 --device 0 --batch-size 32 --data data/dataset_2.yaml  --img 1280 720 --cfg cfg/training/yolov7x.yaml--weights ''--name yolov7-X-model_2 --hyp data/hyp.scratch.p5.yaml --epochs 150 --cache-images

python train.py --workers 8 --device 0 --batch-size 32 --data data/dataset_3.yaml  --img 1280 720 --cfg cfg/training/yolov7x.yaml --weights ''--name yolov7-X-model_2 --hyp data/hyp.scratch.p5.yaml --epochs 150 --cache-images



--save-txt --save-conf --nosave

python detect_custom.py --source datasets/test_data/test/ --weights runs/train/model_1/model_1_last.pt --conf 0.45 --name model_1_last --img-size 1280 --save-txt --save-conf --model 1 --iou-thres 0.3

python detect_custom.py --source datasets/test_data/test/ --weights runs/train/model_2/model_2_last.pt --conf 0.45 --name model_2_last --img-size 1280 --save-txt --save-conf --model 2 --iou-thres 0.3

python detect_custom.py --source datasets/test_data/test/ --weights runs/train/model_3/model_3_last.pt --conf 0.45 --name model_3_last --img-size 1280 --save-txt --save-conf --model 3 --iou-thres 0.3


python detect_custom.py --source datasets/test_data/test/ --weights runs/train/model_1/model_1_best.pt --conf 0.45 --name model_1_best --img-size 1280 --save-txt --save-conf --model 1

python detect_custom.py --source datasets/test_data/test/ --weights runs/train/model_2/model_2_best.pt --conf 0.45 --name model_2_best --img-size 1280 --save-txt --save-conf --model 2

python detect_custom.py --source datasets/test_data/test/ --weights runs/train/model_3/model_3_best.pt --conf 0.45 --name model_3_best --img-size 1280 --save-txt --save-conf --model 3