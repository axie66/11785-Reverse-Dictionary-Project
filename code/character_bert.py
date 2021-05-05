import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from typing import List
from tqdm import tqdm
import sys
sys.path.append('../../code')
import wandb
from dataset import get_data, WantWordsDataset as WWData
from dataset import make_vocab, MaskedDataset, read_json
from models import MaskedRDModel
import datetime
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, BertTokenizer, BertForMaskedLM
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer

# define the model here 

