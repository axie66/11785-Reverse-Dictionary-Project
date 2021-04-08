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


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, ):
        self.lstm = nn.LSTM(*args, **kwargs)
