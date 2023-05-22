CUDA_VISIBLE_DEVICES=4,5,6,7 nohup  python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 train_ris.py \
      --method cls_guide_gt --model lavt --dataset refcoco+ --model_id refcoco+_gt_guide_prompt1_loss5_lr1 --splitBy unc --batch-size 8 --lr 0.00005 \
      --wd 1e-2 --swin_type tiny --pretrained_swin_weights ./pretrained_weights/swin_tiny_patch4_window7_224.pth \
      --NCL 1 --max_tokens 23 --start_guide_epoch -1 --epochs 40 --img_size  480 --resume '' --workers 8  > ./models/refcoco+/output_tiny_gt_guide_prompt1_loss5_lr1 2>&1 &