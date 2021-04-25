import torch
from torch import nn

########################################################################
# Our baseline models:
# - BiLSTM
# - CNN + BiLSTM (?)
# - BiLSTM with Attention
# - BERT
# - Multi-channel Reverse Dictionary Model
# - WantWords
#
# Other models we can try:
# - DistillBERT -> reverse dictionaries for mobile devices?
# - CharacterBERT -> character-level BERT to be more robust to user input
########################################################################

from transformers import BertForMaskedLM

class MaskedRDModel(BertForMaskedLM):
    def set_mask_size(self, mask_size=5):
        self.mask_size = mask_size

    def forward(self, input_ids=None, attention_mask=None, target_matrix=None, 
                      criterion=None, ground_truth=None, **kwargs):
        # input_ids: (batch, def_seq_len)
        # attention_mask: same as input_ids
        # target_matrix: (ww_vocab_size, mask_size), where values are indices into scores matrix
        out = self.bert(input_ids=input_ids, 
                        attention_mask=attention_mask, **kwargs)
        
        # scores: (batch, mask_size, bert_vocab_size) --> contains log probabilities
        scores = self.cls(out[0][:, 1:1+self.mask_size])

        batch_size = scores.shape[0]
        # target_matrix: (batch, mask_size, ww_vocab_size)
        target_matrix = target_matrix.T.unsqueeze(0).expand(batch_size, -1, -1)
        
        # word_scores: (batch, mask_size, ww_vocab_size)
        # For reference: 
        # word_scores[batch][mask_size][ww_vocab_size] = 
        #       scores[batch][mask_size][target_matrix[batch][mask_size][ww_vocab_size]]
        word_scores = torch.gather(scores, 2, target_matrix)
        # (batch, ww_vocab_size)
        # add log probs along mask dim --> equivalent to multiplying probs
        word_scores = torch.sum(word_scores, dim=1) 

        if criterion is not None and ground_truth is not None:
            # ground_truth: (batch,) --> where values range from [0, vocab_size)
            # loss: (batch, ww_vocab_size)
            loss = criterion(word_scores, ground_truth)
            return loss, word_scores
        return word_scores

if __name__ == '__main__':
    from dataset import get_data, MaskedDataset, make_vocab

    model = MaskedRDModel.from_pretrained('bert-base-uncased')
    model.set_mask_size(5)
    x = torch.tensor([[  101,   103,   103,   103,   103,   103,   102,  2008,  2029,  3310,
          2013, 19610,  2361,  1999,  1996,  2832,  1997, 11300, 18809,  2290,
             0,     0],
        [  101,   103,   103,   103,   103,   103,   102,  2000,  3395,  2000,
          2019,  2552,  1997,  2061,  9527,  2100,  2926, 20951,     0,     0,
             0,     0],
        [  101,   103,   103,   103,   103,   103,   102,  2000,  4685, 20302,
          3348,  2588,  1037,  2711,     0,     0,     0,     0,     0,     0,
             0,     0],
        [  101,   103,   103,   103,   103,   103,   102,  3218, 20302,  3348,
          2588,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0],
        [  101,   103,   103,   103,   103,   103,   102,  8872,  9869,  2007,
          2019,  4111,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0],
        [  101,   103,   103,   103,   103,   103,   102,  2383,  2030,  7682,
         22681,  2022, 22155,  2094,  5628,  1037,  2022, 22155,  2094,  3269,
             0,     0],
        [  101,   103,   103,   103,   103,   103,   102, 15525,  1037, 10498,
          2030, 22681,  2019,  2125,  4318,  2022, 22155,  2094, 14894,  2047,
          2259,  2335],
        [  101,   103,   103,   103,   103,   103,   102,  2109,  2926,  1997,
          8288,  7682, 19116, 10732,  6962,  2030, 21995,  1037,  2022, 22155,
          2094, 27940],
        [  101,   103,   103,   103,   103,   103,   102,  3722,  2627,  9049,
          1998,  2627,  2112, 28775, 10814,  1997, 10498,     0,     0,     0,
             0,     0],
        [  101,   103,   103,   103,   103,   103,   102, 19851,  2007, 22681,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0]])

    attention_mask = x > 0

    word_scores = model(input_ids=x, attention_mask=attention_mask)


