import torch
import json

from typing import List
from collections import defaultdict

import torch.nn.utils.rnn as rnn_utils

def get_data(data_dir, word2vec=True):
    print('Loading data...')
    data_dir = f'{data_dir}/%s'

    if word2vec:
        word2vec = read_json(data_dir % 'vec_inuse.json')
        print(f"word2vec: {len(word2vec)} vectors")

    # Train data
    train_data = read_json(data_dir % 'data_train.json')
    train_data_defi = read_json(data_dir % 'data_defi_c.json')
    print(f"Training data: {len(train_data) + len(train_data_defi)} word-def pairs")

    # Dev data
    dev_data = read_json(data_dir % 'data_dev.json')
    print(f"Dev data: {len(dev_data)} word-def pairs")

    # Test data
    # 500 seen words, 500 unseen words, 200 word descriptions (from Hill 2016)
    test_data_seen = read_json(data_dir % 'data_test_500_rand1_seen.json')
    test_data_unseen = read_json(data_dir % 'data_test_500_rand1_unseen.json')
    test_data_desc = read_json(data_dir % 'data_desc_c.json')
    print(f"Test data: {len(test_data_seen) + len(test_data_unseen) + len(test_data_desc)} word-def pairs")

    res = (train_data, train_data_defi, dev_data,
            test_data_seen, test_data_unseen, test_data_desc)
    if word2vec:
        res += (Vectors(word2vec, 300),)
    return res

def read_json(path):
    with open(path) as f:
        return json.load(f)

class Vectors(object):
    '''Simplfied verison of torchtext.vocab.Vectors' class'''
    def __init__(self, embeddings, embedding_dim):
        self.itos = ['<unk>']
        self.itos.extend(list(embeddings.keys()))
        self.stoi = defaultdict(lambda: 0)
        self.embeddings = torch.zeros(len(embeddings)+1, embedding_dim)
        for i, s in enumerate(embeddings):
            self.stoi[s] = i+1
            self.embeddings[i+1,:] = torch.tensor(embeddings[s])

    def get_vecs(self, tokens : List[str]) -> torch.Tensor:
        vecs = [self.embeddings[self.stoi[t]] for t in tokens]
        return torch.stack(vecs)

    def __call__(self, x):
        return self.get_vecs(x)

def make_vocab(data, tokenizer, mask_size=0):
    T = tokenizer.convert_tokens_to_ids
    mask_id = T(['[MASK]'])[0]
    #target2idx: (vocab_size)
    target2idx = {}
    idx2target = []
    # target_matrix: (vocab_size, mask_size)
    target_matrix = [] 
    for dataset in data:
        for entry in dataset:
            target = entry['word']
            if target not in target2idx:
                target_tokens = T(tokenizer.tokenize(target))
                target_pad = mask_size - len(target_tokens)
                # pad/slice target sequences to length mask_size
                if target_pad < 0:
                    target_tokens = target_tokens[:mask_size]
                else:
                    target_tokens.extend([mask_id] * target_pad)
                target2idx[target] = len(target2idx)
                idx2target.append(target)
                target_matrix.append(torch.tensor(target_tokens, dtype=torch.long))
    return torch.stack(target_matrix), target2idx, idx2target

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, definitions, tokenizer, target2idx, 
                 wn_data=None, wn_categories=None, mask_size=0, debug=False):
        super(MaskedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.wn_data = wn_data
        self.wn_categories = wn_categories
    
        self.data = []
        self.mask_size = mask_size

        self.ww_vocab_size = len(target2idx)

        T = tokenizer.convert_tokens_to_ids

        (mask_id, sep_id, cls_id, pad_id, unk_id) = \
            T(['[MASK]', '[SEP]', '[CLS]', '[PAD]', '[UNK]'])

        (self.mask_id, self.sep_id, self.cls_id, self.pad_id, self.unk_id) = \
            (mask_id, sep_id, cls_id, pad_id, unk_id)

        for i, d in enumerate(definitions):
            defn, target = d['definitions'], d['word']
            defn_tokens = tokenizer.tokenize(defn)
            defn_ids = [cls_id] + [mask_id] * mask_size + [sep_id] + T(defn_tokens)
            defn_ids = torch.tensor(defn_ids)

            target_idx = target2idx[target]

            if wn_data is not None and wn_categories is not None:
                wn_ids = [target2idx[target]]
                for cat in wn_categories:
                    wn_ids.extend([target2idx[syn] for syn in 
                             wn_data[target][cat]])
                wn_ids = list(set(wn_ids)) # remove duplicates
                # wn_ids: (ww_vocab_size,)
                wn_ids = torch.sparse_coo_tensor(indices=[wn_ids], 
                            values=torch.tensor(1).expand(len(wn_ids)), 
                            size=(self.ww_vocab_size,))
                elem = (defn_ids, target_idx, wn_ids)
            else:
                elem = (defn_ids, target_idx)
            self.data.append(elem)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_fn(self, batch):
        if self.wn_data is not None:
            Xs = rnn_utils.pad_sequence([x for x,_,_ in batch], padding_value=self.pad_id, batch_first=True)
            Ys = torch.tensor([y for _,y,_ in batch])
            syns = torch.stack([syn for _,_,syn in batch])
            return Xs, Ys, syns
        else:
            Xs = rnn_utils.pad_sequence([x for x,_ in batch], padding_value=self.pad_id, batch_first=True)
            Ys = torch.tensor([y for _,y in batch])
            return Xs, Ys

class WantWordsDataset(torch.utils.data.Dataset):  
    def __init__(self, definition_data, tokenizer, embeddings=None):
        '''
        definition_data: List of dictionaries, where each dictionary contains
                         a definition-word pair (can directly feed a dict
                         returned by the get_data function)
        tokenizer:       Tokenizes string/batch of strings using BPE tokenize.
                         See SentenceBertForRD in models.py for an example.
        embeddings:      (optional) Embedding object (returned by 
                         get_data function)
        '''
        super(WantWordsDataset, self).__init__()
        self.definitions = [(d['definitions'], d['word']) for d in definitions]
        self.tokenizer = tokenizer
        self.embeddings = embeddings

        if embeddings is not None:
            self.stoi = embeddings.stoi
        else:
            unique = {word for _, word in self.definitions}
        
    def __getitem__(self, i):
        return self.definitions[i]

    def __len__(self):
        return len(self.definitions)
    
    def collate_fn(self, batch, word2vec=True):
        Xs = self.tokenizer([x for x, _ in batch], return_tensors='pt', padding=True)
        Xs = (Xs['input_ids'], Xs['attention_mask'])
        Ys = [y for _, y in batch]
        if word2vec and self.embeddings is not None:
            Yvecs = self.embeddings.get_vecs(Ys)
            Yidx = [self.stoi[w] for w in Ys]
            return (Xs, (Yvecs, Ydx))
        else:
            Ys = torch.tensor(Ys)
            return (Xs, Ys)

class Dataset1(torch.utils.data.Dataset):
    def __init__(self, definitions, embeddings, embedding_dim, tokenizer=None):
        super(Dataset1, self).__init__()
        if tokenizer is None:
            tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.tokenizer = tokenizer
        
        self.embedding_dim = embedding_dim
        self.embeddings = embeddings
        
        f = lambda x: not torch.all(embeddings.get_vecs_by_tokens([x]) == 0)
        # Filter out words that do not have embeddings
        self.definitions = definitions.loc[definitions[0].apply(f)]
        self.definitions.index = pd.RangeIndex(len(self.definitions.index))
        
    def __getitem__(self, i):
        word, def_text = self.definitions.loc[i] # definition, in plain text form
        tokens = self.tokenizer(def_text)
        return self.embeddings.get_vecs_by_tokens(tokens), self.embeddings.get_vecs_by_tokens([word]).squeeze()
    
    def __len__(self):
        return len(self.definitions)
    
    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda elem: len(elem[0]), reverse=True)
        Xs = [x for x, _ in batch]
        Ys = ([y for _, y in batch])
        return (torch.nn.utils.rnn.pack_sequence(Xs), 
                torch.stack(Ys))

# def makeDataset(dictionary_path, embedding_path, embedding_dim, tokenizer):
#     dictionary = pd.read_csv(dictionary_path)
#     embeddings = torchtext.vocab.Vectors(embedding_path)
#     return DictDataset(dictionary, embeddings, embedding_dim, tokenizer)

if __name__ == '__main__':
    pass
    # d, v = get_data()
    # dataset = WantWordsDataset(d[1], v, 300, None)
