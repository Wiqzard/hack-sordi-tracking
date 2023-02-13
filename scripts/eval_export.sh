


cd PaddleDetection

WEIGHTS=/home/5qx9nf8a/team_workspace/PaddleDetection/tracking/model_final.pdparams
#configs=/home/5qx9nf8a/team_workspace/PaddleDetection/configs/mot/bytetrack/detector/ppyoloe_plus_m_bytetrack.yml
CONFIGS=/home/5qx9nf8a/team_workspace/PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml 

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${CONFIGS} -o weights=${WEIGHTS} --amp


CONFIGS=/home/5qx9nf8a/team_workspace/PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml 
WEIGHTS=/home/5qx9nf8a/team_workspace/PaddleDetection/tracking/model_final.pdparams

python tools/export_model.py -c ${CONFIGS} -o weights=${WEIGHTS} trt=True
