CUDA_VISIBLE_DEVICES=4,5,6,7 nohup  python -u -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 train_ris.py \
      --method cls_guide_mask --model lavt --dataset refcoco+ --model_id refcoco+_gt_guide_mask --splitBy unc --batch-size 16 --lr 0.00005 \
      --wd 1e-2 --swin_type tiny --pretrained_swin_weights ./pretrained_weights/swin_tiny_patch4_window7_224.pth \
      --NCL 0 --max_tokens 20 --start_guide_epoch -1 --epochs 40 --img_size  480 --resume '' --workers 8  > ./models/refcoco+/output_tiny_gt_guide_mask 2>&1 &