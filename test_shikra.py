max_tokens = 28
import os
# 加载文本
# pre-process the raw sentence
from bert.tokenization_bert import BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# initialize model and load weights
from bert.modeling_bert import BertModel
from lib import segmentation

# pre-process the input image
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np
# inference
import torch.nn.functional as F
import pycocotools.mask as mask_util
import json
def coco_segmentation_to_binary_mask(coco_segmentation):
    # 从COCO格式的分割掩码中提取运行长度编码
    rle_encoded = {
        'size': coco_segmentation['size'],
        'counts': coco_segmentation['counts'].encode('utf-8')
    }

    # 将运行长度编码转换为二进制掩码
    binary_mask = mask_util.decode(rle_encoded)

    return binary_mask

image_transforms = T.Compose(
    [
     T.Resize(480),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

class args:
    swin_type = 'base'
    window12 = True
    mha = ''
    fusion_drop = 0.0
    NCL = 1
    ctx_dim = 768
    save_result = True
    result_dir = '/home/yajie/doctor/RIS/LAVT-RIS/shikra_result'

def predict(image_path, sentence, model, bert_model, device, add_cls=True):
    # 加载图像
    img = Image.open(image_path).convert("RGB")
    img_ndarray = np.array(img)  # (orig_h, orig_w, 3); for visualization
    original_w, original_h = img.size  # PIL .size returns width first and height second
    img = image_transforms(img).unsqueeze(0)  # (1, 3, 480, 480)
    img = img.to(device)  # for inference (input)

    # 去掉graspnet类别名
    sub_index = sentence.find(' X ')
    sentence = sentence[sub_index+3:]
    
    
    sub_index = sentence.find(' X ')
    cls_name = sentence[sub_index+3:]
    sentence = sentence[:sub_index]
    # 计算原句子长度
    sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
    temp_len = len(sentence_tokenized)
    if add_cls:
        # 加classname
        sentence = sentence + ' ' +  ' '.join(["X"] * 1) + ' ' + cls_name
    
    # 编码sentence
    sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
    sentence_tokenized = sentence_tokenized[:max_tokens]  # if the sentence is longer than 20, then this truncates it to 20 words
    # pad the tokenized sentence
    padded_sent_toks = [0] * max_tokens
    padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
    attention_mask = [0] * max_tokens
    attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)
    # convert lists to tensors
    padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)  # (1, 20)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 20)
    padded_sent_toks = padded_sent_toks.to(device)  # for inference (input)
    attention_mask = attention_mask.to(device)  # for inference (input)

    last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
    if add_cls:
        last_hidden_states[0][temp_len-1 : temp_len] = model.ctx
    embedding = last_hidden_states.permute(0, 2, 1)

    output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))
    output = output.argmax(1, keepdim=True)  # (1, 1, 480, 480)
    output = F.interpolate(output.float(), (original_h, original_w))  # 'nearest'; resize to the original image size
    output = output.squeeze()  # (orig_h, orig_w)
    output = output.cpu().data.numpy()  # (orig_h, orig_w)
    output = output.astype(np.uint8)  # (orig_h, orig_w), np.uint8

    if args.save_result:
        # Overlay the mask on the image
        visualization = overlay_davis(img_ndarray, output, sentence)  # red
        scene, _, _, _ = image_path.split('/')[-4:]
        visualization.save(os.path.join(args.result_dir, scene + '_'+ sentence + '.png'))

    return output

def load_model():
    device = 'cuda:1'
    # 加载模型
    single_model = segmentation.__dict__['lavt'](pretrained='', args=args)
    single_model.to(device)
    weights = '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_extract_subject_base_paper_lavt_prompt1_loss1.0_umd_new_no_cls_28_rule/model_best.pth'
    checkpoint = torch.load(weights, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    model_class = BertModel
    single_bert_model = model_class.from_pretrained('bert-base-uncased')
    single_bert_model.pooler = None
    single_bert_model.load_state_dict(checkpoint['bert_model'])
    bert_model = single_bert_model.to(device)

    model.eval()
    bert_model.eval()

    return model, bert_model, device



# show/save results
def overlay_davis(image, mask, sentence, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.5):
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

    # 将sentence放在图片上
    im_overlay = Image.fromarray(im_overlay.astype(image.dtype))
    chars_x, chars_y = 20, 20
    font = ImageFont.truetype("arial.ttf", 
        size=np.floor(3e-2 * 1000 + 0.5).astype('int32')
    )  # 

    img_draw = ImageDraw.Draw(im_overlay)  
    label_size = img_draw.textsize(sentence, font)
    img_draw.rectangle(
        [chars_x, chars_y, chars_x + label_size[0] , chars_y + label_size[1]],
        outline=(0, 50, 128),
        width=1,
        fill=(0, 50, 128)  # 用于填充
    )

    img_draw.text([chars_x, chars_y], sentence, fill=(255, 0, 0), font=font)
    return im_overlay

def load_data_from_jsonl(filename):
    json_data = []
    with open(filename, 'r') as file:
        for line in file:
            item = json.loads(line)
            json_data.append(item)
    
    return json_data

def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

import cv2
if __name__ == '__main__':

    os.makedirs(args.result_dir, exist_ok=True)
    # 加载模型
    model, bert_model, device = load_model()

    file_name = '/home/yajie/doctor/RIS/test/graspnet/REC_garspnet_lavt.jsonl'
    json_data = load_data_from_jsonl(file_name)

    cum_I, cum_U = 0, 0
    seg_total = 0
    mean_IoU = []

    for item in json_data:
        img_path = item['img_path']
        sentence = item['expression']
        coco = item['coco']
        mask = mask_util.decode(coco)
        # mask = mask * 255
        # cv2.imwrite('gt.png', mask.astype(np.uint8))
        pred = predict(img_path, sentence, model, bert_model, device, add_cls=False)

        I, U = computeIoU(pred, mask)
        if U == 0:
            this_iou = 0.0
        else:
            this_iou = I*1.0/U
        mean_IoU.append(this_iou)
        cum_I += I
        cum_U += U

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    print('overall IoU is %.2f\n' % (cum_I * 100. / cum_U))