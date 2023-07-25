# python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcoco_extract_subject_base_paper_prompt1_loss1.0/model_39.pth' \
#         --dataset refcoco --split testA --splitBy unc --swin_type base \
#         --NCL 1 --max_tokens 23 --use_new new

# CUDA_VISIBLE_DEVICES=1  python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_extract_subject_base_paper_lavt_prompt1_loss1.0_umd_new_no_cls_28_WARMUP1_SEED1994/model_best.pth' \
#         --dataset refcocog --split val --splitBy umd --swin_type base \
#         --NCL 1 --max_tokens 28 --use_new new_no_cls --len_thresh 0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcoco+.pth' \
#         --dataset refcoco+ --split testB --splitBy unc --swin_type base \
#         --NCL 0 --max_tokens 20 --use_new none --len_thresh 7

# CUDA_VISIBLE_DEVICES=1 python -u test.py \
#         --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_cost_aggre_fea_base_paper_lavt_prompt0_loss1.0_umd_frz_cls_26_WARMUP1/model_best_ce.pth' \
#         --swin_resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_cost_aggre_fea_base_paper_lavt_prompt0_loss1.0_umd_frz_cls_26_WARMUP1_SEED1994_early_stop_pretrain_none/swin_model.pth' \
#         --cost_aggre --fea_aggre --dataset refclef --split val --splitBy unc --swin_type base \
#         --NCL 0 --max_tokens 26 --use_new frz_cls --len_thresh 0 --seed 7



ckpt=/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_cost_aggre_fea_base_paper_lavt_prompt0_loss1.0_umd_none_20_WARMUP1/model_best.pth
swin_ckpt=/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_cost_aggre_fea_base_paper_lavt_prompt0_loss1.0_umd_frz_cls_26_WARMUP1_SEED1994_early_stop_light/swin_model.pth



CUDA_VISIBLE_DEVICES=1 python -u test.py \
        --resume ${ckpt} \
        --pretrained_swin_weights 'skip' \
        --cost_aggre  --fea_aggre --dataset refclef --split val --splitBy unc --swin_type base \
        --NCL 0 --max_tokens 20 --use_new none --len_thresh 0 --seed 531

# CUDA_VISIBLE_DEVICES=2 python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_cost_aggre_fea_base_paper_lavt_prompt0_loss1.0_umd_none_20_WARMUP1_SEED1994/model_best.pth' \
#         --pretrained_swin_weights /home/yajie/doctor/RIS/LAVT-RIS/pretrained_weights/swin_base_patch4_window12_384_22k.pth \
#         --cost_aggre --fea_aggre --dataset refclef --split val --splitBy unc --swin_type base \
#         --NCL 0 --max_tokens 20 --use_new none --len_thresh 0

# CUDA_VISIBLE_DEVICES=0 python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcoco.pth' \
#         --pretrained_swin_weights /home/yajie/doctor/RIS/LAVT-RIS/pretrained_weights/swin_base_patch4_window12_384_22k.pth \
#         --dataset refcoco --split testB --splitBy unc --swin_type base \
#         --NCL 0 --max_tokens 25 --use_new none --len_thresh 0 ss

#  CUDA_VISIBLE_DEVICES=1 python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/gref_umd.pth' \
#         --dataset refcocog --split val --splitBy umd --swin_type base \
#         --NCL 0 --max_tokens 20 --use_new none --len_thresh 0s

        #--swin_resume ${swin_ckpt} \