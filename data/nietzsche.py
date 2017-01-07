# Kyuhong Shim 2016

import numpy as np
import re

# Nietzsche text data.
# https://s3.amazonaws.com/text-datasets/nietzsche.txt
# Download nietzsche.txt, save as ANSI encoding.
# Change to character-level sequences
# Character sequence to sentences tokenizer from: (D Greenberg)
# http://stackoverflow.com/questions/4576077/python-split-text-on-sentences


def split_into_sentences(text):

    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    digits = "([0-9])"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2",text)
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences if len(s) > 16]
    return sentences


def load_nietzsche(base_datapath):
    nietzsche = open(base_datapath + 'nietzsche/nietzsche.txt')
    corpus = nietzsche.read().lower()
    print('Corpus length: ', len(corpus))

    characters = sorted(list(set(corpus)))
    # print('Total characters: ', len(characters))
    corpus = corpus.replace('\n', ' ')
    corpus = corpus.replace(characters[-4], ' ')
    corpus = corpus.replace(characters[-3], ' ')
    corpus = corpus.replace(characters[-2], ' ')
    corpus = corpus.replace(characters[-1], ' ')
    for i in range(5, 0, -1):
        corpus = corpus.replace(' ' * i, ' ')
    characters = sorted(list(set(corpus)))
    print('Total characters: ', len(characters))
    print(characters)
    sentences = split_into_sentences(corpus)
    print('Total sentences: ', len(sentences))

    return sentences  # return list of sentences


if __name__ == '__main__':
    base_datapath = 'D:/Dropbox/Project/data/'
    load_nietzsche(base_datapath)