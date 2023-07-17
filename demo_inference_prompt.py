# image_path = './demo/ADE_val_00000012.jpg'
# sentence = 'the blue woman'
image_path = '/home/AI-T1/DatasetPublic/RIS/robot/scene_0100/kinect/rgb_crop/0000.png'
# sentence = 'the tail of elephant which is not near the person'
sentence = 'a gold color bottle'
weights = '/home/yajie/doctor/RIS/LAVT-RIS/checkpoints/refcocog_extract_subject_base_paper_lavt_prompt1_loss1.0_umd_new_no_cls_28_rule/model_best.pth'
device = 'cuda:1'

# pre-process the input image
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np

# 加载图像
img = Image.open(image_path).convert("RGB")
img_ndarray = np.array(img)  # (orig_h, orig_w, 3); for visualization
original_w, original_h = img.size  # PIL .size returns width first and height second

image_transforms = T.Compose(
    [
     T.Resize(480),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

img = image_transforms(img).unsqueeze(0)  # (1, 3, 480, 480)
img = img.to(device)  # for inference (input)

# 加载文本
# pre-process the raw sentence
from bert.tokenization_bert import BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_tokens = 28

# 计算原句子长度
sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
temp_len = len(sentence_tokenized)

# 加classname
sentence = sentence + ' ' +  ' '.join(["X"] * 1) + ' ' + 'bottle'
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

# initialize model and load weights
from bert.modeling_bert import BertModel
from lib import segmentation

# construct a mini args class; like from a config file


class args:
    swin_type = 'base'
    window12 = True
    mha = ''
    fusion_drop = 0.0
    NCL = 1
    ctx_dim = 768

# 加载模型
single_model = segmentation.__dict__['lavt'](pretrained='', args=args)
single_model.to(device)
model_class = BertModel
single_bert_model = model_class.from_pretrained('bert-base-uncased')
single_bert_model.pooler = None

checkpoint = torch.load(weights, map_location='cpu')
single_bert_model.load_state_dict(checkpoint['bert_model'])
single_model.load_state_dict(checkpoint['model'])
model = single_model.to(device)
bert_model = single_bert_model.to(device)

model.eval()
bert_model.eval()
# inference
import torch.nn.functional as F
last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
last_hidden_states[0][temp_len-1 : temp_len] = single_model.ctx
embedding = last_hidden_states.permute(0, 2, 1)

output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))
output = output.argmax(1, keepdim=True)  # (1, 1, 480, 480)
output = F.interpolate(output.float(), (original_h, original_w))  # 'nearest'; resize to the original image size
output = output.squeeze()  # (orig_h, orig_w)
output = output.cpu().data.numpy()  # (orig_h, orig_w)


# show/save results
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.5):
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



output = output.astype(np.uint8)  # (orig_h, orig_w), np.uint8
# Overlay the mask on the image
visualization = overlay_davis(img_ndarray, output)  # red

image_name = image_path.split('/')[-1][:-4]
visualization.save('./demo/' + image_name + '_'+ sentence + '.jpg')




