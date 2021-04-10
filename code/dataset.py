import torch
import json

from typing import List
from collections import defaultdict

def get_data():
    print('Loading data...')
    data_dir = '../data/wantwords/%s'

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

    return ((train_data, train_data_defi, dev_data,
            test_data_seen, test_data_unseen, test_data_desc), 
            Vectors(word2vec, 300))

def read_json(path):
    with open(path) as f:
        return json.load(f)

class Vectors(object):
    '''Simplfied verison of torchtext.vocab.Vectors' class'''
    def __init__(self, embeddings, embedding_dim):
        self.itos = list(embeddings.keys())
        self.stoi = defaultdict(lambda: 0)
        self.embeddings = torch.zeros(len(embeddings)+1, embedding_dim)
        for i, s in enumerate(embeddings):
            self.stoi[s] = i+1
            self.embeddings[i+1,:] = torch.tensor(embeddings[s])

    def get_vecs(self, tokens : List[str]) -> torch.Tensor:
        vecs = [self.embeddings[self.stoi[t]] for t in tokens]
        return torch.stack(vecs)

    def __getitem__(self, x):
        return self.get_vecs(x)

class WantWordsDataset(torch.utils.data.Dataset):  
    def __init__(self, definitions, embeddings, embedding_dim, tokenizer):
        super(WantWordsDataset, self).__init__()
        self.definitions = [(d['definitions'], d['word']) for d in definitions]
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        
    def __getitem__(self, i):
        return self.definitions[i]
    
    def __len__(self):
        return len(self.definitions)
    
    def collate_fn(self, batch):
        #batch.sort(key=lambda elem: len(elem[0]), reverse=True)
        Xs = self.tokenizer([x for x, _ in batch], return_tensors='pt', padding=True)
        Ys = self.embeddings.get_vecs([y for _, y in batch])
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
    d, v = get_data()
    dataset = WantWordsDataset(d[1], v, 300, None)