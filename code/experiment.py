import sys
import torch
import numpy as np
import pandas as pd
try:
    import torchtext
except ImportError:
    sys.path.append('/usr/local/lib/python3.8/site-packages/')
    import torchtext

dict_data_path = '../data/dictionary/reverse-dict-singleton.tsv'
glove_embed_dim = 300  # other options are 100, 200, 300
glove_embed_path = f'../data/glove_embed/glove.6B.{glove_embed_dim}d.txt'

# Load dictionary data
# Assuming the .tsv files from Prof Oflazer are
# placed in the data/dictionary folder
dict_data = pd.read_csv(dict_data_path, sep='\t', header=None)
dict_data.head(5)

# Load pretrained GloVe embeddings
# Download them from http://nlp.stanford.edu/data/glove.6B.zip
# and place them in the data/glove_embed folder
glove_embed = torchtext.vocab.Vectors(glove_embed_path)

s, p = glove_embed.get_vecs_by_tokens(['ice', 'gorilla'])
print(s @ p)

d, p = glove_embed.get_vecs_by_tokens(['ice', 'cold'])
print(d @ p)


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, definitions, embeddings, embedding_dim, tokenizer=None):
        super(DictDataset, self).__init__()
        if tokenizer is None:
            tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.embeddings = embeddings

        def f(x):
            return not torch.all(embeddings.get_vecs_by_tokens([x]) == 0)

        # Filter out words that do not have embeddings
        self.definitions = definitions.loc[definitions[0].apply(f)]
        self.definitions.index = pd.RangeIndex(len(self.definitions.index))

    def __getitem__(self, i):
        # definition, in plain text form
        word, def_text = self.definitions.loc[i]
        tokens = self.tokenizer(def_text)
        return self.embeddings.get_vecs_by_tokens(tokens), \
            self.embeddings.get_vecs_by_tokens([word]).squeeze()

    def __len__(self):
        return len(self.definitions)

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda elem: len(elem[0]), reverse=True)
        Xs = [x for x, _ in batch]
        Ys = ([y for _, y in batch])
        return (torch.nn.utils.rnn.pack_sequence(Xs),
                torch.stack(Ys))


data = DictDataset(dict_data, glove_embed, 50)
loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=16,
                                     collate_fn=DictDataset.collate_fn)

# The below code is attempting to learn the word embedding from
# the definition, which isn't exactly what we want to do
model = torch.nn.LSTM(50, 50)
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for i, (x, y) in zip(range(1000), loader):
    optim.zero_grad()
    out, (h, c) = model(x)
    (out_pad, out_lengths) = torch.nn.utils.rnn.pad_packed_sequence(out)
    out_embeds = torch.stack(
        list(out_pad[j] for j in zip(out_lengths-1, range(len(out_lengths)))))
    loss = criterion(out_embeds, y)
    if i % 100 == 0:
        print(i, loss.detach())
    loss.backward()
    optim.step()

out, (h, c) = model(x)
(out_pad, out_lengths) = torch.nn.utils.rnn.pad_packed_sequence(out)

out_pad.shape, out_lengths


for q in list(zip(out_lengths-1, range(len(out_lengths)))):
    r = out_pad[q]
