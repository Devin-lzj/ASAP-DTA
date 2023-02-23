# predict_with_pretrained_model_Human.py
#
# This file is for a quick check of the reported results in the paper.
# It contains replication results only for Human.



from models.ASAP_global_pooling_LE import ASAPNet_GLOBALLE
from models.ASAP_hierarchical_pooling_LE import ASAPNet_HIERLE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
from utils import *


def predicting(model, device, loader):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            predicted_values = torch.sigmoid(output)  # continuous value
            predicted_labels = torch.round(predicted_values)  # convert to binary value

            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # continuous
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # binary
            total_true_labels = torch.cat((total_true_labels, data.y.view(-1, 1).cpu()), 0)

    return total_true_labels.numpy().flatten(), total_pred_values.numpy().flatten(), total_pred_labels.numpy().flatten()


datasets = ['Human']  # only Human here, as the BindingDB dataset has been partitioned beforehand.

modelings = [ASAPNet_HIERLE]     # Here, you need to manually add a trained model, and only one model can be used for each prediction
# modelings = [ASAPNet_GLOBALLE]

Model = modelings[0].__name__
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 512

result = []
valid_results = []
# There are totally five folds in the Human dataset.
for fold in range(1, 6):
    processed_data_file_test = 'data/processed/Human_test_' + str(fold) + '.pt'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run prepare_data_Human.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset='Human_test_' + str(fold))
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        for modeling in modelings:
            model_st = modeling.__name__
            print('\npredicting for the Human dataset using ', model_st)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_file_name = 'pretrained/model_' + model_st + '_Human_fold' + str(fold) + '.model'  # this should also be modified accordingly.
            if os.path.isfile(model_file_name):            
                model.load_state_dict(torch.load(model_file_name), strict=False)
                G, P_value, P_label = predicting(model, device, test_loader)
                tpr, fpr, _ = precision_recall_curve(G, P_value)

                ret = [roc_auc_score(G, P_value), auc(fpr, tpr), precision_score(G, P_label), recall_score(G, P_label)]
                valid_results.append([ret[0], ret[1], ret[2], ret[3]])
                ret = ['Human', model_st] + [round(e, 3) for e in ret]
                result += [ret]
                print('dataset, model, auc, prc, precision, recall')
                print(ret)
            else:
                print('model is not available!')

print(valid_results)
valid_results = np.array(valid_results)
valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]
print("5-fold cross test finished. \n"
      "auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}"
      .format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1], valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))

result_file_name = 'pretrained/Predicted_result_' + Model + '_' + datasets[0] + '.csv'
with open(result_file_name, 'w') as f:
    f.write('dataset,model,auc,prc,precision,recall\n')
    for ret in result:
        f.write(','.join(map(str, ret)) + '\n')
    f.write('5-fold cross test finished. \n')
    f.write("auc:{:.3f}±{:.4f} | prc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}"
         .format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))