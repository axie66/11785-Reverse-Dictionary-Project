import torch
try:
    import torchtext
except ImportError:
    sys.path.append('/usr/local/lib/python3.8/site-packages/')
    import torchtext

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, definitions, embeddings, embedding_dim, tokenizer=None):
        super(DictDataset, self).__init__()
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

def makeDataset(dictionary_path, embedding_path, embedding_dim, tokenizer):
    dictionary = pd.read_csv(dictionary_path)
    embeddings = torchtext.vocab.Vectors(embedding_path)
    return DictDataset(dictionary, embeddings, embedding_dim, tokenizer)