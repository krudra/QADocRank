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
from joint_model import DocumentRanker
from joint_dataloader import DocumentDataset
import pdb
import glob
import random
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
FOLDS = ['fold1','fold2','fold3','fold4','fold5']

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

    print('Complete Fold {} with {} entries'.format(ofname,index))
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


def model_train(net, criterion, opti, train_loader, val_loader, test_loader, dev_data, test_data, args, F):
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
    test_acc, test_loss = inference(net, criterion, test_loader, args, test_data, args.OUTPUT_PATH + F + '_score_' + str(ep) + '.txt')
    t2 = time.time()
    print('Test takes {} secs'.format(t2-t1))
    torch.save(net.state_dict(), args.MODEL_PATH+'{}.dat'.format(F))

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type = int, default = 0)
    #parser.add_argument('-freeze_bert', action='store_true')
    #parser.add_argument('-maxlen', type=int, default=512)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-print_every', type=int, default=500)
    parser.add_argument('-INPUT_PATH', type=str, default='NULL')
    parser.add_argument('-OUTPUT_PATH', type=str, default='NULL')
    parser.add_argument('-MODEL_PATH', type=str, default='NULL')
    parser.add_argument('-max_eps', type=int, default=1)
    args = parser.parse_args()

    #Instantiating the model
    print('Building the model')
    t1 = time.time()

    for F in FOLDS:
    
        net = DocumentRanker()
        net.cuda(args.gpu)
    
        print('Creating criterion and optimizer objects')
        criterion = nn.BCEWithLogitsLoss()
        opti = optim.Adam(net.parameters(), lr=args.lr)

        query_text = {}
        document_text = {}
        
        train = []
        ifname = args.INPUT_PATH + F + '_train.txt'
        fp = open(ifname,'r')
        for i,l in enumerate(fp):
            if i>0:
                wl = l.split('\t')
                t = (wl[1].strip(' \t\n\r'),wl[2].strip(' \t\n\r'),int(wl[0]))
                train.append(t)
                query_text[wl[1].strip(' \t\n\r')] = wl[3].strip(' \t\n\r')
                document_text[wl[2].strip(' \t\n\r')] = wl[4].strip(' \t\n\r')
        fp.close()
        
        valid = []
        ifname = args.INPUT_PATH + F + '_valid.txt'
        fp = open(ifname,'r')
        for i,l in enumerate(fp):
            if i>0:
                wl = l.split('\t')
                t = (wl[1].strip(' \t\n\r'),wl[2].strip(' \t\n\r'),int(wl[0]))
                valid.append(t)
                query_text[wl[1].strip(' \t\n\r')] = wl[3].strip(' \t\n\r')
                document_text[wl[2].strip(' \t\n\r')] = wl[4].strip(' \t\n\r')
        fp.close()
        
        test = []
        ifname = args.INPUT_PATH + F + '_test.txt'
        fp = open(ifname,'r')
        for i,l in enumerate(fp):
            if i>0:
                wl = l.split('\t')
                t = (wl[1].strip(' \t\n\r'),wl[2].strip(' \t\n\r'),int(wl[0]))
                test.append(t)
                query_text[wl[1].strip(' \t\n\r')] = wl[3].strip(' \t\n\r')
                document_text[wl[2].strip(' \t\n\r')] = wl[4].strip(' \t\n\r')
        fp.close()

        #Creating dataholders
        print('Creating train and val dataloaders')
        t1 = time.time()
        train_set = DocumentDataset(train,query_text,document_text,MAXLEN)
        dev_set = DocumentDataset(valid,query_text,document_text,MAXLEN)
        test_set = DocumentDataset(test,query_text,document_text,MAXLEN)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=5)
        val_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=5)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=5)
        print('Done in {} seconds'.format(time.time()-t1))

        print('Let the training begin')
        t1 = time.time()
        model_train(net,criterion,opti,train_loader,val_loader,test_loader,valid,test,args,F)
        print('Done in {} seconds'.format(time.time()-t1))
        print('Done with fold {}'.format(F))
