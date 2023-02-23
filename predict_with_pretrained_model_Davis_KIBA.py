# predict_with_pretrained_model_Davis_KIBA.py
#
# This file is for a quick check of the reported results in the paper.
# It contains replication results for Davis and KIBA datasets.
#
# The pre-trained model for Davis and KIBA is currently not available,
# and you can train the model first before you use this file.
# Other available pre-trained models for check:
# 1. model_ASAPNet_GLOBALLE_BindingDB
# 2. model_ASAPNet_GLOBALLE_Human
# 3. model_ASAPNet_HIERLE_BindingDB
# 4. model_ASAPNet_HIERLE_Human
#

import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.ASAP_hierarchical_pooling_LE import ASAPNet_HIERLE
from models.ASAP_global_pooling_LE import ASAPNet_GLOBALLE
from utils import *

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = ['davis', 'kiba']
modelings = [ASAPNet_HIERLE, ASAPNet_GLOBALLE]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 512

result = []
for dataset in datasets:
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run prepare_data_Davis_KIBA.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        for modeling in modelings:
            model_st = modeling.__name__
            print('\npredicting for ', dataset, ' using ', model_st)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_file_name = 'pretrained/model_' + model_st + '_global_' + dataset + '.model'
            if os.path.isfile(model_file_name):            
                model.load_state_dict(torch.load(model_file_name), strict=False)
                G,P = predicting(model, device, test_loader)
                ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
                ret =[dataset, model_st] + [round(e, 3) for e in ret]
                result += [ ret ]
                print('dataset,model,rmse,mse,pearson,spearman,ci')
                print(ret)
            else:
                print('model is not available!')
with open('result.csv', 'w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')
