#!/usr/bin/python3

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import time
from joint_new_model import DocumentRanker
from joint_dataloader import DocumentDataset
import pdb
import glob
import random, gzip
import pickle, json
from sklearn.metrics import precision_recall_fscore_support

##### SEED SET #####
seed = 123
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
####################

MAXLEN = 512

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze()==labels).float().mean()
    return acc


def inference(net, criterion, dataloader, args, test_data, ofname):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0
    fo = open(ofname,'w')
    index = 0
    with torch.no_grad():
        for query_token, segment_id, attn_mask, labels in dataloader:
            query_token, segment_id, attn_mask, labels = query_token.cuda(args.gpu), segment_id.cuda(args.gpu), attn_mask.cuda(args.gpu), labels.cuda(args.gpu)
            logits = net(query_token, segment_id, attn_mask)
            mean_loss+=criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc+=get_accuracy_from_logits(logits,labels)
            T = logits.to(torch.device("cpu"))
            PRN = T.numpy().tolist()
            probs = torch.sigmoid(logits.unsqueeze(-1))
            P = probs.to(torch.device("cpu"))
            FRN = P.numpy().tolist()
            for i in range(0,len(PRN),1):
                fo.write(str(PRN[i][0])+ '\t' + str(1-FRN[i][0][0]) + '\t' + str(FRN[i][0][0]) + '\n')
                index+=1
            count+=1

    print('Complete Data {} with {} entries'.format(ofname,index))
    return mean_acc/count, mean_loss/count

def evaluate(net, criterion, dataloader, args):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for query_token, segment_id, attn_mask, labels in dataloader:
            query_token, segment_id, attn_mask, labels = query_token.cuda(args.gpu), segment_id.cuda(args.gpu), attn_mask.cuda(args.gpu), labels.cuda(args.gpu)
            logits = net(query_token, segment_id, attn_mask)
            mean_loss+=criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc+=get_accuracy_from_logits(logits,labels)
            count+=1

    return mean_acc/count, mean_loss/count

def model_train(net, criterion, opti, train_loader, test_loader, test_data, args):
    best_acc=0

    for ep in range(args.max_eps):
        net.train()
        for it, (query_token, segment_id, attn_mask, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()

            #converting these to cuda tensors
            query_token, segment_id, attn_mask, labels = query_token.cuda(args.gpu), segment_id.cuda(args.gpu), attn_mask.cuda(args.gpu), labels.cuda(args.gpu)
            
            #Obtaining the logits from the model
            logits = net(query_token, segment_id, attn_mask)

            #Computing loss
            loss = criterion(logits.squeeze(-1),labels.float())

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

            if it%args.print_every==0:
                acc = get_accuracy_from_logits(logits,labels)
                print('Iteration {} of epoch {} complete. Loss: {} Accuracy: {}'.format(it, ep, loss.item(), acc))
    net.eval()
    t1 = time.time()
    test_acc, test_loss = inference(net, criterion, test_loader, args, test_data, args.OUTPUT)
    t2 = time.time()
    print('Test takes {} secs'.format(t2-t1))
    torch.save(net.state_dict(),args.MODEL_NAME)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type = int, default = 1)
    #parser.add_argument('-freeze_bert', action='store_true')
    #parser.add_argument('-maxlen', type=int, default=512)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-print_every', type=int, default=500)
    parser.add_argument('-max_eps', type=int, default=1)
    parser.add_argument('-MODEL_NAME', type=str, default='TREC19.dat')
    parser.add_argument('-TRAIN_INPUT', type=str, default='TREC19_TRAIN.txt.gz')
    parser.add_argument('-TEST_INPUT', type=str, default='TREC19_TEST.txt.gz')
    parser.add_argument('-OUTPUT', type=str, default='TREC19_TEST_RESULT.txt')
    args = parser.parse_args()


    #Instantiating the model
    print('Building the model')
    t1 = time.time()

    net = DocumentRanker()
    net.cuda(args.gpu)
    
    print('Creating criterion and optimizer objects')
    criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(net.parameters(), lr=args.lr)

    query_text = {}
    document_text = {}

    query = {}
    doc = {}

    train = []
    fp = gzip.open(args.TRAIN_INPUT,'rt')
    for i,l in enumerate(fp):
        if i>0:
            wl = l.split('\t')
            t = (wl[1].strip(' \t\n\r'),wl[2].strip(' \t\n\r'),int(wl[0]))
            train.append(t)
            query_text[wl[1].strip(' \t\n\r')] = wl[3].strip(' \t\n\r')
            document_text[wl[2].strip(' \t\n\r')] = wl[4].strip(' \t\n\r')
    fp.close()
        
    test = []
    fp = gzip.open(args.TEST_INPUT,'rt')
    for i,l in enumerate(fp):
        if i>0:
            wl = l.split('\t')
            t = (wl[1].strip(' \t\n\r'),wl[2].strip(' \t\n\r'),int(wl[0]))
            test.append(t)
            query[wl[1].strip(' \t\n\r')] = wl[3].strip(' \t\n\r')
            doc[wl[2].strip(' \t\n\r')] = wl[4].strip(' \t\n\r')
    fp.close()
    
    #Creating dataholders
    print('Creating train and val dataloaders')
    t1 = time.time()
    train_set = DocumentDataset(train,query_text,document_text,MAXLEN)
    test_set = DocumentDataset(test,query,doc,MAXLEN)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=5)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=5)
    print('Done in {} seconds'.format(time.time()-t1))
    
    print('Let the training begin')
    t1 = time.time()
    model_train(net,criterion,opti,train_loader,test_loader,test,args)
    print('Done in {} seconds'.format(time.time()-t1))
