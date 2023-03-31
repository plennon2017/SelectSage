import json
import pickle
import os
from dgl import load_graphs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from e_graph_sage import EGraphSAGE
from loss import FocalLoss, BinaryFocalLoss
from models import MLPBinaryClassifer, MLPMultiClassifer

SCALER_PATH = "./scaler"
CONFIG_DIR = "./config"
LOG_DIR = "./logs"

def get_loss_func(loss_str, num_classes, gamma=5.0):
    if loss_str == "crossentropy":
        if num_classes > 2:
            return nn.CrossEntropyLoss()
        else:
            return F.binary_cross_entropy_with_logits
    elif loss_str == "focalloss":
        if num_classes > 2:
            return FocalLoss(gamma=gamma)
        else:
            return BinaryFocalLoss(gamma=gamma)
    else:
        raise ValueError("Invalid selection of loss. Select either crossentropy or focalloss.")


def train_and_test(graph, features, train_labels, test_labels, valid_labels, in_size, e_size,
                  emb_h_size, mlp_h1_size, mlp_h2_size, num_classes,
                  loss_str, learning_rate=0.001, batch_size=1024, similarity_threshold=0.0, 
                  similarity_function=None, gamma=5.0, name=None, epochs=100):
    # setting a same seed everytime in the beginning will ensure that 
    # the different result is actually due to change in hyperparameters
    # instead of different random initialization of weights
    torch.random.manual_seed(1)
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_DIR, name + '_' + curr_time)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    graph_model = EGraphSAGE(in_size, e_size, emb_h_size, similarity_threshold,
                             similarity_function)
    # creating embedding for all edges using EGraphSAGE algorithm
    print(f'Embeding generation is in progress for configuration {name}')
    with torch.no_grad():
        embedding = graph_model(graph, features)
    print(f'Embedding has been generated successfully for configuration {name}')
    print(f'Saving graph model as {name}_graph_model.pkl')
    graph_model_path = name + '_graph_model_' + curr_time + '.pkl'
    torch.save(graph_model.state_dict(), os.path.join(log_dir, graph_model_path))
    # Embedding for train, valid and test edges of graph
    train_embedding = embedding[graph.edata['train_mask']]
    test_embedding = embedding[graph.edata['test_mask']]
    valid_embedding = embedding[graph.edata['valid_mask']]

    if num_classes > 2:
        model = MLPMultiClassifer(embedding.shape[1], mlp_h1_size, mlp_h2_size, num_classes)
    else:
        model = MLPBinaryClassifer(embedding.shape[1], mlp_h1_size, mlp_h2_size)
    # setting parameters for optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # getting the loss function
    criterian = get_loss_func(loss_str=loss_str, num_classes=num_classes, gamma=gamma)
    accuracy = Accuracy()

    num_batches = int(len(train_embedding) / batch_size)
    losses = []
    valid_accuracy = []
    best_val_acc = 0
    best_epoch = 0
    print(f'Training of classifier model for configuration {name} has started ...')
    for e in range(epochs):
        loss_per_epoch = 0
        for i in range(num_batches):
            batch_train_embedding = train_embedding[i*batch_size : (i+1)*batch_size]
            if num_classes > 2:
                batch_train_labels = train_labels[i*batch_size : (i+1)*batch_size].long()
            else:
                batch_train_labels = train_labels[i*batch_size : (i+1)*batch_size].reshape(-1,1).float()
            
            pred = model.forward(batch_train_embedding)

            if loss_str == "crossentropy":
                loss = criterian(pred, batch_train_labels)
            else:
                loss = criterian.forward(pred, batch_train_labels)
            
            loss_per_epoch += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss_per_epoch / len(train_embedding))
        valid_acc = accuracy(model.predict(valid_embedding), valid_labels)
        valid_accuracy.append(valid_acc)
        print(f'Epoch {e+1} Loss {loss_per_epoch / len(train_embedding)} Accuracy {valid_acc}')

        if valid_acc > best_val_acc:
            print(f'Saving best model at epoch {e+1}')
            best_model_path = name + '_classifier_' + curr_time + '.pkl'
            torch.save(model.state_dict(), os.path.join(log_dir, best_model_path))
            best_val_acc = valid_acc
            best_epoch = e+1
    
    prediction = model.predict(test_embedding)
    test_acc = accuracy(prediction, test_labels)
    f1_score_weighted = f1_score(y_true=test_labels, y_pred=prediction, average='weighted')
    f1_score_micro = f1_score(y_true=test_labels, y_pred=prediction, average='micro')
    confusion_matrix_test = confusion_matrix(y_true=test_labels, y_pred=prediction)
    classification_report_test = classification_report(y_true=test_labels, y_pred=prediction,
                                                       output_dict=True)
    metrics = {'validation_accuracy':best_val_acc, 
               'test_accuracy':test_acc, 'best_epoch':best_epoch,
               'f1_score_weighted':f1_score_weighted, 
               'f1_score_micro':f1_score_micro,
               'confusion_matrix':confusion_matrix_test,
               'classification_report':classification_report_test,
               'loss_history':losses,
               'valid_acc_history':valid_accuracy}

    metrics_path = name + '_metrics_' + curr_time + '.pkl'
    # Saving all metrics as pkl object
    with open(os.path.join(log_dir, metrics_path), 'wb') as newfile:
        pickle.dump(metrics, newfile)


def main(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            hyperparam = json.loads(f.read())
    else:
        raise FileNotFoundError("hyperparam configuration file not found")

    # name = config_path.split(sep="\\")[-1].split(sep=".")[0]
    name = os.path.split(config_path)[-1].split(sep=".")[0]

    if hyperparam["graph_path"]:
        glist, _ = load_graphs(hyperparam["graph_path"])
        graph = glist[0]
    else:
        raise ValueError("Invalid or no path specified for graph")

    # Initializing hyperparameters
    in_size = graph.ndata['feat'][0].shape[0]
    e_size = graph.edata['weight'][0].shape[0]
    # getting features from the graph
    features = graph.ndata['feat']

    if hyperparam["num_classes"] > 2:
        # Defining multiclass labels
        labels = graph.edata['multiclass']
        train_labels = labels[graph.edata['train_mask']]
        test_labels = labels[graph.edata['test_mask']]
        valid_labels = labels[graph.edata['valid_mask']]
    elif hyperparam["num_classes"] == 2:
        # Defining binary labels
        labels = graph.edata['label']
        train_labels = labels[graph.edata['train_mask']]
        test_labels = labels[graph.edata['test_mask']]
        valid_labels = labels[graph.edata['valid_mask']]
    else:
        raise ValueError("Invalid value for num_classes in hyperparam configuration")

    train_and_test(graph=graph, features=features, train_labels=train_labels,
                    test_labels=test_labels, valid_labels=valid_labels,
                    in_size=in_size, e_size=e_size,
                    emb_h_size=hyperparam["emb_h_size"],
                    mlp_h1_size=hyperparam["mlp_h1_size"],
                    mlp_h2_size=hyperparam["mlp_h2_size"], 
                    num_classes=hyperparam["num_classes"],
                    loss_str=hyperparam["loss"],
                    learning_rate=hyperparam["learning_rate"], 
                    batch_size=hyperparam["batch_size"],
                    similarity_threshold=hyperparam["similarity_threshold"],
                    similarity_function=hyperparam["similarity_function"], 
                    gamma=hyperparam["gamma"], name=name,
                    epochs=hyperparam["epochs"])

if __name__ == "__main__":
    config_files = ['multiclass_baseline_sim3_f1.json',
                    'multiclass_baseline_sim3_f5.json']
#                    'multiclass_baseline_sim3_f4.json',
#                    'binaryclass_binary_features 7 205_256_54.json',
#                    'binaryclass_binary_features 5 205_256_54.json',
#                    'binaryclass_binary_features 3 205_256_54.json']
#        'binaryclass_all_features 0 205_256_54.json',
#                    'binaryclass_all_features 5 205_256_54.json',
#                    'binaryclass_mult_features 7_56_256_54.json',
#                    'all_multiclass_baseline.json',
#                    'all_multiclass_baseline_sim3.json',
#                    'all_multiclass_baseline_sim5.json',
#                    'all_multiclass_baseline_sim7.json',
#                    'multiclass_baseline_sim7.json']
#        'all_multiclass_baseline.json',
#                    'all_multiclass_baseline_sim3.json']
#                    'binaryclass_mult_features 0_56_256_54.json',
#					'binaryclass_mult_features 7_56_256_54.json']
 #                   'binaryclass_exploit_features3b.json',
#                   'binaryclass_exploit_features3c.json',
 #                   'binaryclass_exploit_features3d.json',
 #					'binaryclass_exploit_features3e.json',
#					'binaryclass_exploit_features3f.json',
#					'binaryclass_all_features3MLP.json']

    for each_config in config_files:
        config_path = os.path.join(CONFIG_DIR, each_config)
        if not os.path.exists(config_path):
            raise FileNotFoundError("Configuration file {} not found".format(config_path))

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    for each_config in config_files:
        config_path = os.path.join(CONFIG_DIR, each_config)
        print(f'Training for {each_config} in progress')
        main(config_path)