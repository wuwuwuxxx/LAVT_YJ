import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import utils
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')




    
def criterion(output, target, cls_tar, b):

    # cls 的前景像素点较多
    loss1 = nn.functional.cross_entropy(output[b:], cls_tar,  reduction='mean')
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    
    # 后期需增大RIS loss的权重
    weight_mask = 1 + cls_tar
    loss2 = nn.functional.cross_entropy(output[:b], target, weight=weight, reduction='none')

    loss2 = loss2 * weight_mask
    loss2 = loss2[loss2 > 0].mean()

    return 0.5 * loss1 + 5 * loss2

def train_one_epoch(args, model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model, ctx=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions, sentences_len, cls_sen, cls_atten, _, cls_tar = data
        # image,  sentences, attentions = image.cuda(non_blocking=True),\
        #                                     sentences.cuda(non_blocking=True),\
        #                                        attentions.cuda(non_blocking=True)
        image,  sentences, attentions = image.cuda(),\
                                            sentences.cuda(),\
                                               attentions.cuda()
        
        cls_sen, cls_atten = cls_sen.cuda(), cls_atten.cuda()
        cls_sen = cls_sen.squeeze(1)
        cls_atten = cls_atten.squeeze(1)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)


        image = torch.cat([image, image], dim=0)
        sen = torch.cat([sentences, cls_sen])
        atten = torch.cat([attentions, cls_atten])
        B, C, H, W = image.shape
        half_b = int(B / 2)
        target = target.cuda()
        cls_tar = cls_tar.cuda()
        target = F.interpolate(target.float().unsqueeze(1), size=(int(H / 4), int(W / 4)), mode='nearest').long().squeeze(1)
        cls_tar = F.interpolate(cls_tar.float().unsqueeze(1), size=(int(H / 4), int(W / 4)), mode='nearest').long().squeeze(1)


        if bert_model is not None:
            last_hidden_states = bert_model(sen, attention_mask=atten)[0]  # (6, 10, 768)
            if args.NCL > 0:
                for i in range(half_b):  
                    temp_len = sentences_len[i] + 1
                    if (temp_len + args.NCL) <= args.max_tokens:
                        # print(temp_len)
                        last_hidden_states[i][temp_len: (temp_len + args.NCL)] = ctx

            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            atten = atten.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, l_mask=atten)
        else:
            output = model(image, sentences, l_mask=attentions)

        loss = criterion(output, target, cls_tar, half_b)

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        # del image, target, sentences, attentions, loss, output, data
        # if bert_model is not None:
        #     del last_hidden_states, embedding

        # gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        # break



