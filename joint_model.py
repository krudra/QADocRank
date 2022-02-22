import torch
import torch.nn as nn
from transformers import BertModel
import random
import numpy as np

##### SEED SET #####
seed = 123
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
####################

class DocumentRanker(nn.Module):
    def __init__(self):
        super(DocumentRanker, self).__init__()

        #Instantiate
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=True)

        #Freeze bert layers
        #if freeze_bert:
        #    for p in self.bert_layer.parameters():
        #        p.requires_grad = False

        
        #Final layer
        self.classification = nn.Linear(768,1)
        

    def forward(self, seq, segment_id, attn_masks):
        '''
        Inputs:
            -seq: Tensors of shape [B,T] containing token ids of sequences
            -attn_masks: Tensor of shape [B,T] containing attention masks to avoid PADDED tokens
        '''

        #Feeding the input to BERT model
        cls_rep = self.bert(seq, token_type_ids=segment_id, attention_mask=attn_masks)['last_hidden_state'][:, 0]

        #Feeding cls_rep to the classifier layer
        logits = self.classification(cls_rep)

        return logits
