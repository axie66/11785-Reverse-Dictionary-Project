###############################################################################
# get_synonyms.py
#
# Simple script to collect certain wordnet data for the words 
# in the WantWords dataset:
# - synonyms
# - antonyms
# - derivationally related forms
# - hyponyms
# - hypernyms
#
###############################################################################


import json
from dataset import get_data, make_vocab
from nltk.corpus import wordnet as wn

d = get_data('../wantwords-english-baseline/data', word2vec=False)

train_data, train_data_def, dev_data, test_data_seen, \
    test_data_unseen, test_data_desc = d

vocab = dict()
for dataset in d:
    for entry in dataset:
        vocab[entry['word']] = None

for word in vocab:
    synonyms = set()
    antonyms = set()
    related_forms = set() # words that can be derived from word
    hyponyms = set() # more general
    hypernyms = set() # more specific
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            l = lemma.name().lower()
            if l in vocab and l != word:
                synonyms.add(l)
            for antonym in lemma.antonyms():
                a = antonym.name().lower()
                if a in vocab and a != word:
                    antonyms.add(a)
            for form in lemma.derivationally_related_forms():
                f = form.name().lower()
                if f in vocab and f != word:
                    related_forms.add(f)
        for hyponym in synset.hyponyms():
            for lemma in hyponym.lemmas():
                l = lemma.name().lower()
                if l in vocab and l != word:
                    hyponyms.add(l)
        for hypernym in synset.hypernyms():
            for lemma in hypernym.lemmas():
                l = lemma.name().lower()
                if l in vocab and l != word:
                    hypernyms.add(l)
    vocab[word] = {
        'synonyms': list(synonyms),
        'antonyms': list(antonyms),
        'related_forms': list(related_forms),
        'hyponyms': list(hyponyms),
        'hypernyms': list(hypernyms)
    }

with open('all_synonyms.json', 'w+') as f:
    json.dump(vocab, f, indent=4)