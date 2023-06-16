# python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcoco_extract_subject_base_paper_prompt1_loss1.0/model_39.pth' \
#         --dataset refcoco --split testA --splitBy unc --swin_type base \
#         --NCL 1 --max_tokens 23 --use_new new

python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_extract_subject_base_paper_lavt_prompt1_loss1.0_umd_new_no_cls_28_rule/model_best.pth' \
        --dataset refcocog --split val --splitBy google --swin_type base \
        --NCL 1 --max_tokens 28 --use_new new --len_thresh 0

# python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog.pth' \
#         --dataset refcoco+ --split testA --splitBy unc --swin_type base \
#         --NCL 0 --max_tokens 20 --use_new none --len_thresh 0

# python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcoco_extract_subject_base_paper_lavt_prompt1_loss1.0_unc_new_no_cls_23_rule3_4/model_best.pth' \
#         --dataset refcoco --split testB --splitBy unc --swin_type base \
#         --NCL 1 --max_tokens 23 --use_new new_no_cls


# python test.py --resume '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/gref_umd.pth' \
#         --dataset refcocog --split val --splitBy google --swin_type base \
#         --NCL 0 --max_tokens 25 --use_new none --len_thresh 0