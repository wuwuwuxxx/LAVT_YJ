import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='LAVT training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--dataset', default='refcocog', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt', help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default='lavt', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='/home/AI-T1/DatasetPublic/RIS/refer/data', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--split', default='train', help='only used when testing')
    parser.add_argument('--splitBy', default='google', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.', default=True)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument('--ctx_dim', default=768, type=int, help='the dim of prompt')
    parser.add_argument('--NCL', default=1, type=int, help='the length of prompt')
    parser.add_argument('--max_tokens', default=23, type=int, help='max tokens')
    parser.add_argument('--start_guide_epoch', default=0, type=int, help='start cls guide epoch')
    parser.add_argument('--momentum', default=0.99, type=float, help='model momentum')
    parser.add_argument('--update_model_frequency', default=5, type=int, help='model update frequency')
    parser.add_argument('--loss_weight', default='1.0', type=float, help='loss weight for expression')
    parser.add_argument('--classifer_lr', default='10.0', type=float, help='classifer lr weight')
    

    parser.add_argument('--method', default='paper', type=str, help='method use')
    parser.add_argument('--use_new', default='new_no_cls', type=str, help='method use')
    parser.add_argument('--len_thresh', default=0, type=int, help='classifer lr weight')

    # seed
    parser.add_argument("--seed", default=531, type=int, help="min learning rate for polylr")

    # lr
    parser.add_argument("--lr", default=0.00005, type=float, help="initial learning rate")
    parser.add_argument("--min_lr", default=0, type=float, help="min learning rate for polylr")
    parser.add_argument("--lr_scheduler", default='polylr', type=str, help="initial learning rate")
    parser.add_argument("--lr-warmup-epochs", default=1, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
