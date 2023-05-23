$DATASET = refcoco+
$SWIN_TYPE = tiny
$METHOD = class_guide_gt
$NCL = 1
$LOSS_WEIGHT = 1.0



CUDA_VISIBLE_DEVICES=0,1,2,3 nohup  python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 train_ris.py \
      --loss_weight ${LOSS_WEIGHT} --classifer_lr 10.0 --method ${METHOD} --model lavt --dataset ${DATASET} --model_id ${DATASET}_${SWIN_TYPE}_${METHOD}_prompt${NCL}_loss${LOSS_WEIGHT}  --splitBy unc --batch-size 8 --lr 0.00005 \
      --wd 1e-2 --swin_type ${SWIN_TYPE} --pretrained_swin_weights ./pretrained_weights/swin_tiny_patch4_window7_224.pth \
      --NCL ${NCL} --max_tokens 23 --start_guide_epoch -1 --epochs 40 --img_size  480 --resume '' --workers 8  > ./models/${DATASET}/output_${SWIN_TYPE}_${METHOD}_prompt${NCL}_loss${LOSS_WEIGHT} 2>&1 &