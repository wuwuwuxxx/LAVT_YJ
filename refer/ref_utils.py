### 提取主语
import spacy

from spacy import displacy
from spacy.tokens import Token, Doc
from typing import Iterable

def load_nlp(): 
    nlp = spacy.load("en_core_web_sm")
    return nlp

# 提取主语
def paper(doc: Doc) -> Token:
    for token in doc:
        if token.dep_ == "ROOT":
            if token.pos_ == "NOUN":
                return token
            elif 'VB' in token.tag_:
                for child in token.children:
                    if child.pos_ == "NOUN":
                        return child

# 判断主语是否和给的类别一致
from nltk.corpus import wordnet2021 as wn
def consistent_with_cls(word1, word2):
    if word1 == word2:
        return True
    # 只看词语的第一定义
    synsets1 = wn.synsets(word1)[:1]
    synsets2 = wn.synsets(word2)[:1]

    lowest_common_subsumer = None
    max_depth = -1

    for synset1 in synsets1:
        for synset2 in synsets2:
            path = synset1.lowest_common_hypernyms(synset2)
            if path:
                depth = path[0].min_depth()
                if depth > max_depth:
                    max_depth = depth
                    lowest_common_subsumer = path[0]

    if lowest_common_subsumer is None:
        return False
    if word1 in lowest_common_subsumer.name():
        return True
    else:
        return False

import Levenshtein as lev    
# 根据类别提取主语
def cls_noun(cls, sentence):

    if cls in sentence:
        return cls
    raw_sentence = sentence
    sentence = sentence.split()
    for item in sentence:
        if item in cls.split() and len(item) >2:
            return item
        if consistent_with_cls(cls, item):
            return item 
        if lev.distance(item, cls) < 2 and len(cls) > 2:
            print(item+', ', cls+', ', sentence)
            return cls
        if lev.distance(item, cls) < 3 and len(cls) > 5:
            print(item+', ', cls+', ', sentence)
            return cls
        
    if cls != 'person':

     
        # 交通工具
        if  cls in ['car', 'bicycle', 'truck', 'motorcycle', 'train', 'boat', 'bus', 'surfboard'] :
            for item in sentence:
                if item in ['van', 'car', 'truck', 'bike', 'bikes', 'truck', 'cart', 'trailer', 'bmw', 'wagon', 'vehicle',  'suv', 'boat', 'ship', 'chevrolet', 
                           'toyota', 'honda', 'yacht', 'bicycle' ,'motorcycle', 'train', 'jeep', 'skis', 'trolley', 'tram', 'bus', 
                           'scooter', 'ambulance', 'crane', 'dirtbike', 'surfboard', 'ford', 'skateboard', 'sail', 'tractor', 'board']:
                    return item
        # 飞机
        if cls in ['airplane']:
            for item in sentence:
                if item in ['aircraft','helicopter', 'flight', 'plain']:
                    return item
        # 甜点
        if cls in ['donut', 'cake', 'sandwich', 'pizza', 'hot dog']:
            if 'ice cream' in raw_sentence:
                return 'ice cream'
            if 'hot dog' in raw_sentence:
                return 'hot dog'
            for item in sentence:
                if item in ['doughnut', 'cookie', 'donut', 'cake', 'bread', 'sandwich', 'pizza', 'cupcake', 'pastry', 'dessert', 'peach', 
                            'food', 'chocolate', 'muffin', 'sausage', 'pie', 'fish', 'burger', 'hotdog', 'pumpkin', 'loaf', 'salad', 'toast',
                            'carrots', 'carrot', 'taco', 'tacos', 'waffles', 'waffle', 'bagel']:
                    return item
        if cls in ['donut', 'cake', 'sandwich', 'pizza', 'hot dog']:
            for item in sentence:
                if item in ['piece', 'slice']:
                    return item
        if cls in ['orange', 'apple', 'banana']:
            for item in sentence:
                if item in ['lemon', 'lemos', 'lime', 'limes', 'fruit', 'fruits','peaches', 'orange', 'apple', 'slice', 'food', 'slices']:
                    return item
        if cls in ['carrot', 'broccoli']:
            if 'french fry' in raw_sentence:
                return 'french fry'
            for item in sentence:
                if item in ['carrot', 'carrots', 'broccoli', 'potatoes', 'cauliflower', 'vegetable', 'vegetables', 'cauli', 'fruit', 'fries', 'veggie']:
                    return item
            for item in sentence:
                if item in ['slice', 'piece']:
                    return item
        
        
        # 牛马羊 
        if cls in ['sheep', 'cow', 'horse', 'giraffe', 'zebra', 'bear', 'elephant']:
            for item in sentence:
                if item in ['sheep', 'cow', 'lamb', 'goat', 'bull', 'animal', 'animals', 'buffalo', 'ram',
                             'bulls', 'donkey', 'foal', 'giraffe', 'zebra', 'horn', 'deer', 'yak', 'donkeys', 'horse', 'trunk', 'butt']:
                    return item
                
        if cls in ['teddy bear', 'cat', 'dog']:
            for item in sentence:
                if item in ['teddybears', 'bears', 'bear', 'animal', 'animals', 'toy', 'doll', 'kola', 'bulls', 'donkey', 'cat', 
                            'animal', 'dog', 'monkey', 'puppy', 'lamb', 'panda', 'pig', 'toys', 'rabbit', 'kitten', 'cow', 'kitty', 'snoopy']:
                    return item
                
        if cls in ['bird']:
            for item in sentence:
                if item in ['chicken', 'crow', 'pegeon', 'ostrich', 'peacock', 'animal', 'bird']:
                    return item
        # 椅子
        if cls in ['chair', 'bench', 'sofa', 'couch']:
            for item in sentence:
                if item in ['chair', 'bench', 'sofa', 'desk', 'stool', 'couch', 'armchair', 'seat', 'loveseat']:
                    return item 
        # 包
        if cls in ['backpack', 'handbag', 'suitcase']:
            for item in sentence:
                if item in ['bag', 'suitcase', 'backpack', 'luggage', 'purse', 'box', 'handbag',]:
                    return item 
        # 电子产品
        if cls in ['tv', 'laptop', 'keyboard']:
            for item in sentence:
                if item in ['laptop', 'computer', 'screen', 'monitor', 'tv', 'keyboard', 'ipad', 'television', 'imac', 'tablet', 'device', 'macbook', 'desktop']:
                    return item
        # 容器     
        if cls in ['cup', 'bowl', 'bottle', 'vase', 'wine glass']:
            if 'wine glass' in raw_sentence:
                return 'wine glass'
            for item in sentence:
                if item in ['container', 'glass', 'mug', 'cup', 'bowl', 'plate', 'dish', 'canister', 'jar', 'pan', 
                            'pitcher', 'pot', 'tray', 'food', 'bucket', 'drink', 'basket', 'holder', 'box', 
                            'bottle','yogurt', 'can', 'coffee', 'vase', 'carrot','jug' , 'carrots', 'glasses', 'peaches',
                            'salad', 'beer', 'fries', 'water', 'cups', 'juice']:
                    return item
        # 植物
        if cls in ['potted plant']:
            for item in sentence:
                if item in ['flower', 'flowers', 'tree', 'trees','houseplant', 'houseplants', 'plants', 'bush', 'roses', 'rose', 'vase', 'jar', 'planter', 'leaf', 'leaves', 'plant', 'pot']:
                    return item
        # 滑雪板
        if cls in ['snowboard']:
            for item in sentence:
                if item in ['snowboard', 'ski', 'skis', 'board']:
                    return item
        if cls in ['toilet', 'sink']:
            for item in sentence:
                if item in  ['urinal', 'urinals', 'toilet', 'sink', 'bathtub', 'pisser']:
                    return item
        # 微波炉
        if cls in ['oven', 'microwave']:
            for item in sentence:
                if item in ['stove', 'dishwasher', 'machine', 'microwave', 'oven', 'burners', 'burner']:
                    return item
        # 手机
        if cls in ['cell phone', 'remote']:
            if 'cell phone' in raw_sentence:
                return 'cell phone'
            for item in sentence:
                if item in ['phone', 'tablet', 'ipad', 'mobile', 'controller', 'laptop', 'iphone', 'camera', 'remote', 'device']:
                    return item
        # 书
        if cls in ['book']:
            for item in sentence:
                if item in ['magazine', 'magazines', 'paper', 'papers', 'newspaper', 'book']:
                    return item
        # 餐具
        if cls in ['spoon', 'fork', 'knife']:
            for item in sentence:
                if item in ['fork', 'paper', 'newspaper', 'ladle', 'server', 'sword', 'knife', 'scooper', 'spoon', 'tool']:
                    return item
        # 桌子
        if cls in ['dining table', 'bed']:
            if 'place mat' in raw_sentence:
                return 'placemat'
            for item in sentence:
                if item in ['table', 'dish', 'plate', 'plates','pizza', 'tray', 'dessert', 'tabletop', 'placemat', 'knife', 'countertop',
                             'desk', 'pillow', 'pillows', 'bowl', 'bed', 'blanket', 'platter', 'cup', 'bunk', 'glasses', 'orange', 'carpet', 
                             'dishes', 'food', 'counter', 'tablecloth', 'bottle', 'matress']:
                    return item
                
        if cls in ['traffic light', 'parking meter']:
            for item in sentence:
                if item in ['sign', 'light', 'lights', 'signal', 'stoplight', 'meter']:
                    return item

    else:
        if 'female' in sentence:
            return 'female'
        if 'male' in sentence:
            return 'male'
        for item in sentence:
            if item in ['hand', 'hands', 'foot', 'hat', 'fireman', 'dress', 'clothing', 'arm', 'shoulder', 'driver', 
                        'shirt', 'head', 'thumb', 'sneakers', 'jacket', 'people', 'finger', 'legs', 'leg', 'pants',
                          'men', 'skiier', 'hair', 'coat', 'sweatshirt', 'face', 'shoe', 'arms', 'jeans', 'shorts', 
                          'scarf', 'police', 'hoodie', 'jersey', 'suit', 'clothes', 'socks', 'fingers', 'shoes', 'sweater',
                          'tshirt', 'uniform', 'skirt', 'woman', 'helmet']:
                return item

    return None



def cal_sim(nlp, cls, doc):
    token_cls = nlp(cls)
    best_sim = 0
    best_words = ''
    
    if len(doc) == 1:
        return doc[0].text
    for token in doc:
        if token.pos_ == 'NOUN':
            temp_sim = token.similarity(token_cls)
            if temp_sim > best_sim:
                best_sim = temp_sim
                best_words = token.text
    if best_sim > 0.75:
        return best_words
    else:
        return None        

if __name__ == '__main__':
    cls = 'person'
    sentence = 'skiier in red pants'
    

    item = cls_noun(cls, sentence)
    print(item)

        
            