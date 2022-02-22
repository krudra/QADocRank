import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import math, sys
import random

##### SEED SET #####
seed = 123
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
####################

class DocumentDataset(Dataset):
    def __init__(self, query_set, query_text, document_text, maxlen):

        #Store the contents of the file in a pandas dataframe
        self.query_set = query_set
        #self.df = pd.read_csv(filename, delimiter='\t')

        self.query_text = query_text
        self.document_text = document_text
        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        
        return len(self.query_set)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        qd_pair = self.query_set[index]
        label = qd_pair[2]
        query_text = self.query_text[qd_pair[0]]
        document_text = self.document_text[qd_pair[1]]
        
        #Preprocessing the text to be suitable for BERT

        query_tokens = self.tokenizer.tokenize(query_text) #Tokenize the sentence
        doc_tokens = self.tokenizer.tokenize(document_text)
        tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
        seg_query_ids = [0 for _ in range(len(query_tokens))]
        LENGTH_SECOND_LIST = self.maxlen - len(query_tokens) - 3
        seg_doc_ids = [1 for _ in range(LENGTH_SECOND_LIST)]


        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices
        tokens_ids_tensor = torch.tensor(tokens_ids)

        segment_ids = self.tokenizer.create_token_type_ids_from_sequences(seg_query_ids,seg_doc_ids)
        segment_ids_tensor = torch.tensor(segment_ids)

        attn_masks = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, segment_ids_tensor, attn_masks, label
