import torch
import torch.utils.data
from torch import nn
import utils
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')








def criterion(output, target, weight_mask):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    loss = nn.functional.cross_entropy(output, target, weight=weight, reduction='none')
    B, H, W = target.shape
    loss = (loss * weight_mask).mean()
    return loss
    

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
        image, target, sentences, attentions, sentences_len, cls_sen, cls_atten = data
        image,  sentences, attentions = image.cuda(non_blocking=True),\
                                            sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)
        
        cls_sen, cls_atten = cls_sen.cuda(non_blocking=True), cls_atten.cuda(non_blocking=True)


        cls_sen = cls_sen.squeeze(1)
        cls_atten = cls_atten.squeeze(1)
        if epoch > args.start_guide_epoch:
            with torch.no_grad():
                last_hidden_states_cls = bert_model(cls_sen, attention_mask=cls_atten)[0]
                embedding_cls = last_hidden_states_cls.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
                attentions_cls = cls_atten.unsqueeze(dim=-1)  # (batch, N_l, 1)
                cls_output = model(image, embedding_cls, l_mask=attentions_cls)

        target = F.interpolate(target.float().unsqueeze(1), size=(int(image.shape[-1] / 4), int(image.shape[-1] / 4)), mode='nearest').long().squeeze().cuda()
        weight_mask = torch.ones_like(target, device=target.device)
        if epoch > args.start_guide_epoch:
            weight_mask[cls_output.argmax(1) == 1] = 2.
            weight_mask[target == 1] = 2.

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (6, 10, 768)
            if args.NCL > 0:
                for i in range(last_hidden_states.shape[0]):  
                    temp_len = sentences_len[i] + 1
                    if (temp_len + args.NCL) <= args.max_tokens:
                        # print(temp_len)
                        last_hidden_states[i][temp_len: (temp_len + args.NCL)] = ctx

            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            attentions = attentions.unsqueeze(dim=-1)  # (batch, N_l, 1)
            output = model(image, embedding, l_mask=attentions)
        else:
            output = model(image, sentences, l_mask=attentions)

        loss = args.loss_weight * criterion(output, target, weight_mask)
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




