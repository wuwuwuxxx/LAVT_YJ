import os
import sys
sys.path.append('/home/yajie/doctor/RIS/LAVT-RIS/')
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER

from args import get_parser

import torch.nn.functional as F

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()
import pickle

class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy, args.use_new)

        self.classes = self.refer.Cats

        self.max_tokens = args.max_tokens

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.sentence_len = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.input_cls_ids = []
        self.cls_atten_masks = []

        self.eval_mode = eval_mode

        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency


        # f_all = open(args.dataset + ' ' + self.split + '_'+ 'all_sent.txt', 'w')

        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []
            sentence_len = []

            # add cls embedding
            cls_for_ref = []
            cls_atten_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):

                # f_all.write(sentence_raw_sent + '\n')
                # continue
                if args.use_new == 'new':
                    sentence_raw = el['raw']
                    sentence_raw_sent = el['sent']
                    # 原sentence长度
                    temp_len = len(self.tokenizer.encode(text=sentence_raw_sent, add_special_tokens=True)) - 2
                    # 找到主语
                    sub_index = sentence_raw.find(' X ')
                    subject = sentence_raw[(sub_index + 3):]
                    # 句子太长对句子进行截取
                    if temp_len > self.max_tokens:
                        sentence_raw_sent = ' '.join(sentence_raw_sent.split(' ')[:self.max_tokens - args.NCL - 2])
                        temp_len = len(self.tokenizer.encode(text=sentence_raw_sent, add_special_tokens=True)) - 2
                    sentence_len.append(temp_len)
                    # 加上主语
                    sentence_raw_sent = sentence_raw_sent + ' ' +  ' '.join(["X"] * args.NCL) + ' ' + subject
                    sentence_raw = sentence_raw_sent
                    # print(sentence_raw)
                elif args.use_new == 'new_no_cls':
                    # 原sentence长度
                    sentence_raw = el['sent']
                    sentence_raw_sent = el['raw']
                    temp_len = len(self.tokenizer.encode(text=sentence_raw_sent, add_special_tokens=True)) - 2
                    # print(temp_len)
                    if temp_len > args.len_thresh:
                        # 找到主语
                        sub_index = sentence_raw.find(' X ')
                        if sub_index > 0:
                            subject = sentence_raw[(sub_index + 3):]
                  
                        if sub_index == -1:
                            subject = ''
                            if temp_len > self.max_tokens:
                                sentence_raw_sent = ' '.join(sentence_raw_sent.split(' ')[:self.max_tokens - args.NCL - 2])
                                temp_len = len(self.tokenizer.encode(text=sentence_raw_sent, add_special_tokens=True)) - 2
                            # # 没有主语就不加X
                            temp_len = -temp_len

                            # print(sentence_raw + ' '+ self.classes[ref['category_id']])
                        else:
                            # 句子太长对句子进行截取
                            if temp_len > self.max_tokens:
                                sentence_raw_sent = ' '.join(sentence_raw_sent.split(' ')[:self.max_tokens - args.NCL - 2])
                                temp_len = len(self.tokenizer.encode(text=sentence_raw_sent, add_special_tokens=True)) - 2
                            # subject = subject.lower()
                            # 加上主语
                            sentence_raw_sent = sentence_raw_sent + ' ' +  ' '.join(["X"] * args.NCL) + ' ' + subject
                    sentence_raw = sentence_raw_sent
                    # print(sentence_raw)
                    # f_all.write(sentence_raw_sent + '\n')
                    # continue
                    el['sent'] =  sentence_raw_sent
                    sentence_len.append(temp_len)

                else:
                    sentence_raw = el['raw']
                    sentence_raw_sent = el['sent']

                    ## add text prompt
                    if args.NCL > 0: 
                        temp_len = len(self.tokenizer.encode(text=sentence_raw_sent, add_special_tokens=True)) - 2
                        sentence_len.append(temp_len)
                        if temp_len > self.max_tokens:
                            # print(sentence_raw)
                            sentence_raw = ' '.join(sentence_raw.split(' ')[:self.max_tokens - args.NCL - 2])
                            # print(sentence_raw)
                        sentence_raw =  sentence_raw + ' ' +  ' '.join(["X"] * args.NCL) + ' ' + self.classes[ref['category_id']]
                    else:
                        temp_len = len(self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)) - 2
                        sentence_len.append(temp_len)

                    # sentence_raw = sentence_raw_sent
                    # print(sentence_raw)

                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)
                # print(len(input_ids))
                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))


                input_ids_cls = self.tokenizer.encode(text='the ' + self.classes[ref['category_id']], add_special_tokens=True)
                cls_attention_mask = [0] * self.max_tokens
                cls_padded_input_ids = [0] * self.max_tokens
                cls_padded_input_ids[:len(input_ids_cls)] = input_ids_cls
                cls_attention_mask[:len(input_ids_cls)] = [1] * len(input_ids_cls)

                cls_for_ref.append(torch.tensor(cls_padded_input_ids).unsqueeze(0))
                cls_atten_for_ref.append(torch.tensor(cls_attention_mask).unsqueeze(0))
                

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
            self.sentence_len.append(sentence_len)
            self.input_cls_ids.append(cls_for_ref)
            self.cls_atten_masks.append(cls_atten_for_ref)

        # f_all.close()
        # with open('subject_dict_' + args.dataset + '.pkl', 'wb') as f_s:
        #     pickle.dump(subject_dict, f_s)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        pimg = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'].split('_')[-1])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        cls_annot = self.refer.getclsMask(ref[0], ref_mask)

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(pimg, annot)
            cls_img, cls_target = self.image_transforms(pimg, cls_annot)

        if self.eval_mode:
            embedding = []
            att = []
            slen = []

            cls_embedding = []
            cls_att = []

            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
                l = torch.tensor(self.sentence_len[index][s]).unsqueeze(0).unsqueeze(0)
                slen.append(l.unsqueeze(-1))

                ce = self.input_cls_ids[index][s]
                ca = self.cls_atten_masks[index][s]
                cls_embedding.append(ce.unsqueeze(-1))
                cls_att.append(ca.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
            sentence_len = torch.cat(slen, dim=-1)

            cls_tensor_embedding = torch.cat(cls_embedding, dim=-1)
            cls_atten_embedding = torch.cat(cls_att, dim=-1)


        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]
            sentence_len = self.sentence_len[index][choice_sent]
            
            cls_tensor_embedding = self.input_cls_ids[index][choice_sent]
            cls_atten_embedding = self.cls_atten_masks[index][choice_sent]
                      


        return img, target, tensor_embeddings, attention_mask, sentence_len, cls_tensor_embedding, cls_atten_embedding, cls_img, cls_target

import transforms as T
if __name__ == '__main__':
    
    # args = get_parser()
    
    def get_transform(args):
        transforms = [T.Resize(args.img_size, args.img_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]

        return T.Compose(transforms)
    
    tfm = get_transform(args)


    rdataset = ReferDataset(args,
                      split='train',
                      image_transforms=tfm,
                      target_transforms=None,
                      )
    
    data_loader = torch.utils.data.DataLoader(rdataset, batch_size=args.batch_size, num_workers=0, pin_memory=args.pin_mem, drop_last=True)

    for item in data_loader:
        pass