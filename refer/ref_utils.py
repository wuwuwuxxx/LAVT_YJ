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