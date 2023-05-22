
# train refcoco
# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py \
#       --model lavt --dataset refcoco --model_id refcoco_prompt --batch-size 12 --lr 0.00005 \
#       --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
#       --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco/output_prompt

# train refcoco+
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 train.py \
      --model lavt --dataset refcoco+ --model_id refcoco+_tiny_480_prompt --splitBy unc --batch-size 8 --lr 0.00005 \
      --wd 1e-2 --swin_type tiny --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
      --epochs 40 --img_size 480 --resume '' 2>&1 | tee ./models/refcoco+/output_prompt

# train refcoco cls guide
# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node 4 --master_port 12347 train_cls_guide.py \
#       --model lavt --dataset refcoco+ --model_id refcoco+_prompt_cls_guide --batch-size 12 --lr 0.00005 \
#       --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
#       --NCL 4 --max_tokens 25 --epochs 40 --img_size 480 --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/model_best_refcoco+_prompt_cls_guide_1.pth' 2>&1 | tee ./models/refcoco+/output_prompt_cls_guide

# train refcoco cls guide momentum
# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node 4 --master_port 12347 train_cls_guide_momentum.py \
#       --model lavt --dataset refcocog --model_id refcocog_cls_guide_momentum --batch-size 8 --lr 0.00005 \
#       --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
#       --NCL 0 --max_tokens 20 --start_guide_epoch -1 --epochs 40 --img_size  480 --resume '' --workers 8 2>&1 | tee ./models/refcocog/output_cls_guide_momentum


# # train refcoco cls guide momentum 3090
# CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 --master_port 12347 train_cls_guide_momentum.py \
#       --model lavt --dataset refcoco+ --model_id refcoco+_prompt_cls_guide_momentum --batch-size 6 --lr 0.00005 \
#       --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
#       --NCL 4 --max_tokens 25 --start_guide_epoch -1 --epochs 40 --img_size  480 --resume '' --workers 4 2>&1 | tee ./models/refcoco+/output_prompt_cls_guide_momentum_3090


# # train refcoco cls guide gt
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 --master_port 12348 train_cls_guide_gt.py \
#       --model lavt --dataset refcoco+ --model_id refcoco+_cls_guide_gt_8 --splitBy unc --batch-size 6 --lr 0.00005 \
#       --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
#       --NCL 0 --max_tokens 20 --start_guide_epoch -1 --epochs 40 --img_size  480 --resume '' --workers 8 