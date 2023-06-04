import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation

import transforms as T
import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict


import warnings
warnings.filterwarnings('ignore')

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert_cls import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=(image_set != 'train'),
                      )
    num_classes = 2

    return ds, num_classes


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def evaluate(model, data_loader, bert_model, ctx=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'


    Refiou = utils.RefIou()
    

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, sentences_len, _, _, _, _ = data
            # image, target, sentences, attentions = image.cuda(non_blocking=True),\
            #                                        target.cuda(non_blocking=True),\
            #                                        sentences.cuda(non_blocking=True),\
            #                                        attentions.cuda(non_blocking=True)
            image, target, sentences, attentions = image.cuda(),\
                                                   target.cuda(),\
                                                   sentences.cuda(),\
                                                   attentions.cuda()           


            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            sentences_len = sentences_len.squeeze(1)

            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    if args.NCL > 0:
                        for i in range(sentences_len.size(0)):  
                            temp_len = sentences_len[i][0][j] + 1
                            if (temp_len + args.NCL) <= args.max_tokens:
                            # print(temp_len)
                                last_hidden_states[i][temp_len: (temp_len + args.NCL)] = ctx

                    embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                    temp_attentions = attentions[:, :, j].unsqueeze(dim=-1)  # (B, N_l, 1)
                    output = model(image, embedding, l_mask=temp_attentions)
                else:
                    output = model(image, sentences, l_mask=attentions)

                # output = F.interpolate(output, size=(480, 480), mode='bilinear', align_corners=True)

                Refiou.update(output.cpu(), target.cpu())

            # break

        Refiou.reduce_from_all_processes()
        print(Refiou)

    return Refiou.over_iou






def main(args):


    if args.method == 'cls_guide_gt':
        from train_cls_guide_gt import train_one_epoch, criterion
    elif args.method == 'cls_guide':
        from train_cls_guide import train_one_epoch, criterion
    elif args.method == 'cls_guide_mask':
        from train_cls_guide_gt_mask import train_one_epoch, criterion
    elif args.method == 'paper':
        from train_paper import train_one_epoch, criterion


    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model.cuda()
    single_model = model
    ctx = None
    if args.NCL > 0:
        ctx = single_model.ctx
    # single_model.ctx.requires_grad = True
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    if args.model != 'lavt_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # a work-around for a bug in Transformers = 3.0.2 that appears for DistributedDataParallel
        bert_model.cuda()
        single_bert_model = bert_model
        if args.distributed:
            bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
            bert_model = torch.nn.parallel.DistributedDataParallel(bert_model, device_ids=[args.local_rank])
    else:
        bert_model = None
        single_bert_model = None

    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'lavt_one':
        if args.NCL > 0:
            params_to_optimize = [
                {'params': backbone_no_decay, 'weight_decay': 0.0, 'lr': args.lr},
                {'params': backbone_decay, 'lr': args.lr},
                {"params": [p for p in single_model.classifier.parameters() if p.requires_grad], 'lr': args.lr * args.classifer_lr},
                # the following are the parameters of bert
                {"params": reduce(operator.concat,
                                [[p for p in single_bert_model.encoder.layer[i].parameters()
                                    if p.requires_grad] for i in range(10)]), 'lr': args.lr},
                # the following are the parameters of ctx
                {
                    "params": [single_model.ctx], 'weight_decay': 0.0, 'lr': args.lr * args.classifer_lr
                },
            ]
        else:
            params_to_optimize = [
                {'params': backbone_no_decay, 'weight_decay': 0.0, 'lr': args.lr},
                {'params': backbone_decay, 'lr': args.lr},
                {"params": [p for p in single_model.classifier.parameters() if p.requires_grad], 'lr': args.lr * args.classifer_lr},
                # the following are the parameters of bert
                {"params": reduce(operator.concat,
                                [[p for p in single_bert_model.encoder.layer[i].parameters()
                                    if p.requires_grad] for i in range(10)]), 'lr': args.lr},
            ]           

    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999


    checkpoint_dir = os.path.join(args.output_dir, args.model_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)
        train_one_epoch(args, model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, bert_model, ctx)
        # iou, overallIoU = evaluate(model, data_loader_test, bert_model, ctx)
        overallIoU = evaluate(model, data_loader_test, bert_model, ctx)
        # print('Average object IoU {}'.format(iou))
        # print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        # save_checkpoint = True
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {'model': single_model.state_dict(), 'bert_model': single_bert_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}
            else:
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict()}

            utils.save_on_master(dict_to_save, os.path.join(checkpoint_dir,
                                                            'model_best.pth'.format(epoch)))
            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
