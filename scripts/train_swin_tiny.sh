# # train refcoco+
# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py \
#       --model lavt --dataset refcoco+ --model_id refcoco+_prompt_tiny_NCL4 --batch-size 8 --lr 0.00005 \
#       --wd 1e-2 --swin_type tiny --pretrained_swin_weights /home/yajie/doctor/RIS/LAVT-RIS/pretrained_weights/swin_tiny_patch4_window7_224.pth \
#       --NCL 4 --max_tokens 22 --epochs 40 --img_size 384 --resume '' 2>&1 | tee ./models/refcoco+/output_prompt_tiny_NCL4



# CUDA_VISIBLE_DEVICES=4,5,6,7  python -m torch.distributed.launch --nproc_per_node 4 --master_port 12347 train.py \
#       --model lavt --dataset refcoco+ --model_id refcoco+_prompt_tiny_NCL1 --batch-size 8 --lr 0.00005 \
#       --wd 1e-2 --swin_type tiny --pretrained_swin_weights /home/yajie/doctor/RIS/LAVT-RIS/pretrained_weights/swin_tiny_patch4_window7_224.pth \
#       --NCL 1 --max_tokens 22 --epochs 40 --img_size 384 --resume '' 2>&1 | tee ./models/refcoco+/output_prompt_tiny_NCL1





# CUDA_VISIBLE_DEVICES=2,3,4,5  python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py \
#       --model lavt --dataset refcoco+ --model_id refcoco+_prompt_tiny --batch-size 8 --lr 0.00005 \
#       --wd 1e-2 --swin_type tiny --pretrained_swin_weights /home/yajie/doctor/RIS/LAVT-RIS/pretrained_weights/swin_tiny_patch4_window7_224.pth \
#       --NCL 0 --max_tokens 22 --epochs 40 --img_size 384 --resume '' 2>&1 | tee ./models/refcoco+/output_prompt_tiny

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup  python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 12348 train_cls_guide_gt.py \
#       --model lavt --dataset refcoco+ --model_id refcoco+_cls_guide_gt_tiny --splitBy unc --batch-size 8 --lr 0.00005 \
#       --wd 1e-2 --swin_type tiny --pretrained_swin_weights ./pretrained_weights/swin_tiny_patch4_window7_224.pth \
#       --NCL 0 --max_tokens 20 --start_guide_epoch -1 --epochs 40 --img_size  480 --resume '' --workers 8  > ./models/refcoco+/output_tiny_gt_guide 2>&1 &


DATASET=refcocog
SWIN_TYPE=base
METHOD=paper
NCL=0
MAX_TOKENS=26
LOSS_WEIGHT=1.0
BRANCH=cost_aggre_fea
MODEL=lavt
USE_NEW=frz_cls
SPLIT=umd
LEN_THRESH=0
WARMUP=1
SEED=1994
NAME=${BRANCH}_${SWIN_TYPE}_${METHOD}_${MODEL}_prompt${NCL}_loss${LOSS_WEIGHT}_${SPLIT}_${USE_NEW}_${MAX_TOKENS}_WARMUP${WARMUP}_SEED${SEED}

CUDA_VISIBLE_DEVICES=0,1,2,3  nohup python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 12348 train_ris.py \
      --cost_aggre --fea_aggre \
      --loss_weight ${LOSS_WEIGHT} \
      --classifer_lr 1.0 \
      --method ${METHOD} \
      --model ${MODEL} \
      --dataset ${DATASET} \
      --model_id ${DATASET}_${NAME} \
      --splitBy ${SPLIT} \
      --batch-size 8 \
      --wd 1e-2 \
      --use_new ${USE_NEW} \
      --swin_type ${SWIN_TYPE} \
      --pretrained_swin_weights /home/yajie/doctor/RIS/LAVT-RIS/pretrained_weights/swin_base_patch4_window12_384_22k.pth \
      --NCL ${NCL} --max_tokens ${MAX_TOKENS} \
      --epochs 40 --img_size 480 \
      --resume '' --workers 8   \
      --lr 0.00005 --lr-warmup-epochs ${WARMUP} \
      --len_thresh ${LEN_THRESH} \
      --seed ${SEED} > ./models/${DATASET}/output_${NAME}  2>&1 &



