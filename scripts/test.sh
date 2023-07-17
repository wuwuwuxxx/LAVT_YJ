# python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcoco_extract_subject_base_paper_prompt1_loss1.0/model_39.pth' \
#         --dataset refcoco --split testA --splitBy unc --swin_type base \
#         --NCL 1 --max_tokens 23 --use_new new

CUDA_VISIBLE_DEVICES=5  python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_extract_subject_base_paper_lavt_prompt1_loss1.0_umd_new_no_cls_28_WARMUP1_SEED1994/model_best.pth' \
        --dataset refclef --split test --splitBy unc --swin_type base \
        --NCL 1 --max_tokens 28 --use_new new_no_cls --len_thresh 0

# CUDA_VISIBLE_DEVICES=4,5,6,7 python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcoco+.pth' \
#         --dataset refcoco+ --split testB --splitBy unc --swin_type base \
#         --NCL 0 --max_tokens 20 --use_new none --len_thresh 7

# python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcoco_extract_subject_base_paper_lavt_prompt1_loss1.0_unc_new_no_cls_23_rule3_4/model_best.pth' \
#         --dataset refcoco --split testB --splitBy unc --swin_type base \
#         --NCL 1 --max_tokens 23 --use_new new_no_cls


#  CUDA_VISIBLE_DEVICES=4 python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/gref_umd.pth' \
#         --dataset refclef --split test --splitBy unc --swin_type base \
#         --NCL 0 --max_tokens 20 --use_new none --len_thresh 0