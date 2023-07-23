import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F

save_result = False
save_dir = './result/'
os.makedirs(save_dir, exist_ok=True)

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert_cls import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes

# show/save results
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)

def scale_img_back(data,output_gpu=True,device=torch.device(0)):
    tmp = data.clone().permute(0,2,3,1)
    if output_gpu:
        for x in tmp:
            x *= torch.FloatTensor([0.229,0.224,0.225]).cuda(device=device)
            x += torch.FloatTensor([0.485,0.456,0.406]).cuda(device=device)
    else:
        for x in tmp:
            x *= torch.FloatTensor([0.229,0.224,0.225])
            x += torch.FloatTensor([0.485,0.456,0.406])

    # return tmp.permute(0,3,1,2)
    return tmp

def evaluate(swin_model, model, data_loader, bert_model, device, dataset_test, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    swin_model.eval()
    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    f_sent = open('low_sent_paper.txt', 'w')
    num_short, num_long = 0, 0
    mean_sIou, mean_lIou = [], []
    cum_s_I, cum_s_U = 0, 0
    cum_l_I, cum_l_U = 0, 0
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, sentences_len, _, _, _, _, index= data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()

            sentences_len = sentences_len.squeeze(1)
            g_fea = swin_model(image)

            input_shape = image.shape[-2:]
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    temp_len = sentences_len[0][0][j] + 1
                    # if  abs(temp_len - 1) <= args.len_thresh:
                    #     continue
                    if args.NCL > 0:
                        for i in range(sentences_len.size(0)):  
                            # temp_len = sentences_len[i][0][j] + 1
                            if (temp_len + args.NCL) <= args.max_tokens and temp_len > args.len_thresh + 1:
                            # print(temp_len)
                                last_hidden_states[i][temp_len: (temp_len + args.NCL)] = model.ctx

                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1), g_fea=g_fea)
                    # output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=True)
                else:
                    output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                if abs(temp_len - 1) > args.len_thresh:
                    mean_lIou.append(this_iou)
                    num_long += 1
                else:
                    mean_sIou.append(this_iou)
                    num_short += 1
                if save_result:
                    if this_iou >= 0 and abs(temp_len - 1) > args.len_thresh:
                        # this_image = scale_img_back(image) * 255
                        # this_image = this_image.cpu().numpy().squeeze().astype(np.uint8)
                        # result = overlay_davis(this_image, output_mask.squeeze())
                        # result_gt = overlay_davis(this_image, target.squeeze())
                        # result = Image.fromarray(result)
                        # result_gt = Image.fromarray(result_gt)

                        this_ref_id = dataset_test.ref_ids[index]
                        this_img_id = dataset_test.refer.getImgIds(this_ref_id)
                        this_img = dataset_test.refer.Imgs[this_img_id[0]]
                        image_name = this_img['file_name']
                        this_sentences = dataset_test.refer.Refs[this_ref_id]['sentences'][j]['sent']
                        f_sent.write(image_name + '\t' +  this_sentences + '\t' +  str(this_iou) + '\n')

                        # result_name = image_name[:-4] + '_'.join(this_sentences.replace('/', '').split(' ')) + '.png'
                        # resultgt_name = image_name[:-4] + '_'.join(this_sentences.replace('/', '').split(' ')) + 'gt.png'
                        # result.save(os.path.join(save_dir, result_name))
                        # result_gt.save(os.path.join(save_dir, resultgt_name))
                        
                        
                cum_I += I
                cum_U += U
                if abs(temp_len - 1) > args.len_thresh:
                    cum_l_I += I
                    cum_l_U += U
                else:
                    cum_s_I += I
                    cum_s_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            # del image, target, sentences, attentions, output, output_mask
            # if bert_model is not None:
            #     del last_hidden_states, embedding

    # f_sent.close()
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)

    # results_str += 'SHORT overall IoU = %.2f\n' % (cum_s_I * 100. / cum_s_U)
    # mean_sIou = np.mean(np.array(mean_sIou))
    # results_str += 'SHORT mean IoU = %.2f\n' % (mean_sIou * 100.)
    results_str += 'LONG overall IoU = %.2f\n' % (cum_l_I * 100. / cum_l_U)
    mean_lIou = np.mean(np.array(mean_lIou))
    results_str += 'LONG mean IoU = %.2f\n' % (mean_lIou * 100.)

    print(results_str)
    print('short: ', num_short)
    print('long: ', num_long)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=True)
    model = single_model.to(device)

    # swin model for embedding guide  只用来提取特征  不做梯度回传
    swin_model = segmentation.__dict__['swin'](pretrained=args.pretrained_swin_weights,
                                              args=args)
    swin_model.cuda()


    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines

        single_bert_model.pooler = None
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(swin_model, model, data_loader_test, bert_model, device, dataset_test, args)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
