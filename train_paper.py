

import torch
import torch.utils.data
from torch import nn



import utils


import torch.nn.functional as F



def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)





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
        image, target, sentences, attentions, sentences_len, _, _, _, _ = data
        # image,  sentences, attentions = image.cuda(non_blocking=True),\
        #                                     sentences.cuda(non_blocking=True),\
        #                                        attentions.cuda(non_blocking=True)
        
        image,  sentences, attentions = image.cuda(),\
                                            sentences.cuda(),\
                                               attentions.cuda()
        target = target.cuda()
        # target = F.interpolate(target.float().unsqueeze(1), size=(int(image.shape[-1] / 4), int(image.shape[-1] / 4)), mode='nearest').long().squeeze()

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
            # output = model(image, sentences, l_mask=attentions)
            if args.distributed:
                last_hidden_states = model.module.text_encoder(sentences, attention_mask=attentions)[0]
            else:
                last_hidden_states = model.text_encoder(sentences, attention_mask=attentions)[0]
            if args.NCL > 0:
                for i in range(last_hidden_states.shape[0]):  
                    temp_len = sentences_len[i] + 1
                    if (temp_len + args.NCL) <= args.max_tokens:
                    # print(temp_len)
                        last_hidden_states[i][temp_len: (temp_len + args.NCL)] = ctx

            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            temp_attentions = attentions.unsqueeze(dim=-1)  # (B, N_l, 1)
            output = model(image, embedding, l_mask=temp_attentions)



        loss = args.loss_weight * criterion(output, target)
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





    


