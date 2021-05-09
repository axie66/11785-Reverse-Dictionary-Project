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
# - SenseBERT -> utilize sense-level information about word
# - GlossBERT -> takes advantage of WordNet gloss data of words
########################################################################

import torch
from torch import nn
from transformers import BertForMaskedLM
from sentence_transformers import SentenceTransformer
import sys
sys.path.append('../character-bert')
from modeling.character_bert import CharacterBertModel

class SentenceBERTForRD(nn.Module):
    def __init__(self, pretrained_name, out_dim, *sbert_args, 
                 freeze_sbert=True, criterion=None, **sbert_kwargs):
        '''
        To use this model, you will need to first run "pip install sentence-transformers"

        Should be used in conjunction with the WantWordsDataset class, i.e.:
        >>> model = SentenceBERTForRD(...)
        >>> dataset = WantWordsDataset(definitions, embeddings, model.tokenizer)

        pretrained_name: Name of pretrained SentenceBERT variant to be used
        vocab_size:      Size of output vocabulary
        freeze_sbert:    Can optionally freeze SentenceBERT model and train
                         only output MLP
        criterion:       (optional) Must be one of CrossEntropyLoss, MSELoss, 
                         and CosineSimilarity
        '''
        super(SentenceBERTForRD, self).__init__()
        self.sbert = SentenceTransformer(pretrained_name, *sbert_args, **sbert_kwargs)
        self.pretrained_name = pretrained_name
        self.freeze_sbert = freeze_sbert
        if freeze_sbert:
            for param in self.sbert.parameters():
                param.requires_grad = False
        
        hidden_dim = self.sbert.get_sentence_embedding_dimension()
        # Simple MLP decoder --> modeled off of BERT MLM head
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

        self.criterion = criterion
        self.classification = None
        if criterion is not None:
            if isinstance(criterion, nn.CrossEntropyLoss):
                self.classification = True
            elif isinstance(criterion, (nn.MSELoss, nn.CosineSimilarity)):
                self.classification = False
            else:
                raise Exception("Criterion must be one of CrossEntropyLoss, MSELoss, or CosineSimilarity")

        # init weights of linear layer
        for layer in self.decoder.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)
                
    def unfreeze(self):
        for param in self.sbert.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, ground_truth=None):
        # embed: (batch, 768)
        embed = self.sbert({
            'input_ids': input_ids, 'attention_mask': attention_mask
        })['sentence_embedding']
        # out: (batch, vocab_size) 
        # prob distribution over vocabulary
        out = self.decoder(embed)

        if self.criterion is not None and ground_truth is not None:
            loss = self.criterion(out, ground_truth)
            return loss, out
        return out

class MaskedRDModel(BertForMaskedLM):
    def initialize(self, mask_start=1, mask_size=5, multilabel=False, ww_vocab_size=0, pos_weight=5):
        self.mask_start = mask_start
        self.mask_size = mask_size

        if multilabel:
            self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(pos_weight).expand(ww_vocab_size))
        else:
            self.xent_criterion = nn.CrossEntropyLoss()
            # Learned transformation for each mask sequence
            # self.W_prob = nn.Parameter(torch.ones(1, ww_vocab_size, 1, mask_size),
            #                            requires_grad=True)

    def forward(self, input_ids=None, attention_mask=None, 
                      target_matrix=None, ground_truth=None, sep_id=102,
                      wn_ids=None, weight_gt=10, **kwargs):
        # input_ids: (batch, def_seq_len)
        # attention_mask: same as input_ids
        # target_matrix: (ww_vocab_size, mask_size), where values are indices into scores matrix
        # wn_ids: (batch, ww_vocab_size)
        # Note: can assume that the sep token will not be located at end of seq
        sep_locations = torch.roll(input_ids == sep_id, shifts=1, dims=-1)
        sep_locations[:, 0] = 0 # last [SEP] will wrap to 0th position
        token_type_ids = (torch.cumsum(sep_locations, dim=-1) > 0).long()
        
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, **kwargs)
        
        # scores: (batch, mask_size, bert_vocab_size) --> contains log probabilities
        scores = self.cls(out[0][:, self.mask_start:self.mask_start+self.mask_size])

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

        if wn_ids is not None and ground_truth is not None:
            # wn_ids: (batch, ww_vocab_size) (sparse tensor)
            # loss: (batch, ww_vocab_size)
            loss = self.bce_criterion(word_scores, wn_ids)
            # weight ground truth labels more heavily
            gt_loss = torch.gather(loss, 1, ground_truth.unsqueeze(1)) * weight_gt
            # gt_loss: (batch, 1)
            loss = loss.scatter(1, ground_truth.unsqueeze(1), gt_loss)
            loss = loss.sum() / batch_size
            return loss, word_scores
        elif ground_truth is not None:
            # ground_truth: (batch,) --> where values range from [0, vocab_size)
            # loss: (batch, ww_vocab_size)
            loss = self.xent_criterion(word_scores, ground_truth)
            return loss, word_scores
        return word_scores

class CharacterBERTForRD(nn.Module):
    def __init__(self, vocab_size, *cbert_args, freeze_cbert=True, criterion=nn.CrossEntropyLoss(), **cbert_kwargs):
        super(CharacterBERTForRD, self).__init__()
        self.cbert = CharacterBertModel.from_pretrained('../character-bert/pretrained-models/general_character_bert')
        self.freeze_cbert = freeze_cbert
        if freeze_cbert:
            for param in self.cbert.parameters():
                param.requires_grad = False
        
        # Simple MLP decoder --> modeled off of BERT MLM head
        self.decoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Linear(768, vocab_size),
        )

        self.criterion = criterion

        # init weights of linear layer
        for layer in self.decoder.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)
                
    def unfreeze(self):
        for param in self.cbert.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, ground_truth=None):
        embed, _ = self.cbert(input_ids)
        # out: (batch, sentence_length, 768) 
        # prob distribution over vocabulary
        out = self.decoder(embed[:, 1])

        if self.criterion is not None and ground_truth is not None:
            loss = self.criterion(out, ground_truth)
            return loss, out
        return out

