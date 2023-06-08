"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import sys
sys.path.append('/home/yajie/doctor/RIS/LAVT-RIS/')
import os.path as osp
import json
import pickle as pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np 
from pycocotools import mask
from PIL import Image



class REFER:

    def __init__(self, data_root, dataset='refcoco', splitBy='unc', use_new = 'none', save_change=True):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        if dataset == 'refcocog':
            print('Split by {}!'.format(splitBy))
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
            self.IMAGE_DIR = '/home/AI-T1/DatasetPublic/robseg/SSeg_datasets/coco/train2017'
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        if use_new == 'new':
            ref_file = osp.join(self.DATA_DIR, 'new_refs(' + splitBy + ').p')
        elif use_new == 'new_no_cls':
            ref_file = osp.join(self.DATA_DIR, 'new_rule_2_refs(' + splitBy + ').p')
        else:
            ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        
        print('ref_file:', ref_file)
        self.data = {}
        self.data['dataset'] = dataset
        f = open(ref_file, 'r')
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))


            
        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # extract subject
        if save_change:
            from ref_utils import load_nlp, paper,  consistent_with_cls
            self.nlp = load_nlp()
            self.saveChangeRef()

            with open(osp.join(self.DATA_DIR, 'new_rule_2_refs(' + splitBy + ').p'), 'wb') as f_new:
                pickle.dump(self.data['refs'], f_new)
                
                

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')


    def saveChangeRef(self):
        from ref_utils import load_nlp, paper,  consistent_with_cls, cls_noun, cal_sim
        print('save change...')
        # fetch info from instances
        Cats = {}
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # with open('right_' + self.data['dataset'] + '.pkl', 'rb') as f_r:
        #     right = pickle.load(f_r)
        # right = []
        with open('no_subject_' + self.data['dataset'] + '.txt', 'w') as f:
            for j, ref in enumerate(self.data['refs']):
                # if j in right:
                #     continue
                category_id = ref['category_id']
                # 类别信息
                cls_name = Cats[category_id]
                # add mapping of sent
                for i, sent in enumerate(ref['sentences']):
                    target_info = ' X '
                    sentence = sent['sent']
                    doc = self.nlp(sentence)
                    # 提取到的主语
                    subject = paper(doc)
                    # 最后的主语
                    sub_item = None
                    if subject is not None:
                        relation = consistent_with_cls(cls_name, subject.text)
                        if relation:
                            # right.append(j)
                            sub_item = subject.text
                            # target_info = target_info + subject.text
                    # 用规则提取
                    if sub_item is None:
                        if cls_name in sentence:
                            sub_item = cls_name
                        else:
                            sub_item = cls_noun(cls_name, sentence)
                    # 用相似度提取
                    if sub_item is None:
                        sub_item = cal_sim(self.nlp, cls_name, doc)
                        # if sub_item is None:
                        #     f.write(sentence + '/' + cls_name +'\n')
                    # 用spacy再提取
                    if sub_item is None:
                        # sub_item = paper(doc)
                        # if sub_item is not None:
                        #     sub_item = sub_item.text
                        # else:
                        sub_item = 'none'
                        f.write(sentence + '/' + cls_name +'\n')
                    
                    target_info = target_info + sub_item
                    sent['raw'] = sentence + target_info
            # print(ref['sentences'])
        # with open('right_' + self.data['dataset'] + '.pkl', 'wb') as f_r:
        #     pickle.dump(right, f_r)
        print(len(self.data['refs']))
                
                





    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if
                         image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]

        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']

        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {'mask': m, 'area': area}


    def getclsMask(self, ref, ref_mask):
        
        image_id = ref['image_id']
        annos = self.imgToAnns[image_id]
        cat_id = ref['category_id']
        image = self.Imgs[ref['image_id']]
        
        cls_m = np.zeros(ref_mask.shape)
        for ann in annos:
            if ann['category_id'] == cat_id:# and ann['id'] != ref['ann_id']:
                if  'counts' in ann['segmentation']:
                    continue
                if  type(ann['segmentation'][0]) == list:  # polygon
                    rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
                else:
                    rle = ann['segmentation']

                m = mask.decode(rle)
                m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)
                cls_m[m==1] = 1
        annot = Image.fromarray(cls_m.astype(np.uint8), mode="P")
        return annot

    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


if __name__ == '__main__':

    data_root = '/home/AI-T1/DatasetPublic/RIS/refer/data'
    refer = REFER(data_root, dataset='refcoco+', splitBy='unc')
    ref_ids = refer.getRefIds()

    ref_ids = refer.getRefIds(split='train')
    print('There are %s training referred objects.' % len(ref_ids))

    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if len(ref['sentences']) < 2:
            continue
        print('The label is %s.' % refer.Cats[ref['category_id']])
        plt.figure()
        refer.showRef(ref, seg_box='box')
        plt.show()

