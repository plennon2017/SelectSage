import json
import os
from dgl import load_graphs
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from datetime import datetime
from sklearn.metrics import f1_score
from e_graph_sage import EGraphSAGE
from loss import FocalLoss, BinaryFocalLoss
from models import MLPBinaryClassifer, MLPMultiClassifer

HYPERPARAM_CONFIG = "hyperparam_tuning_config.json"
RESULT_PATH = "./hyperparam_tuning_results"

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


def train_and_test(graph, features, train_labels, test_labels, valid_labels,
          emb_h_size, mlp_h1_size, mlp_h2_size, num_classes, 
          loss_str, learning_rate=0.001, batch_size=1024, similarity_threshold=0.0, 
          similarity_function=None, gamma=5.0):
    # setting a same seed everytime in the beginning will ensure that 
    # the different result is actually due to change in hyperparameters
    # instead of different random initialization of weights
    torch.random.manual_seed(1)
    graph_model = EGraphSAGE(in_size, e_size, emb_h_size, similarity_threshold,
                             similarity_function)
    # creating embedding for all edges using EGraphSAGE algorithm
    print("Embeding generation is in progress")
    with torch.no_grad():
        embedding = graph_model(graph, features)
    print("Embedding as been generated successfully")
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
    print("Training of classifier model has started ...")
    for e in range(hyperparam["epochs"]):
        loss_per_epoch = 0
        for i in range(num_batches):
            batch_train_embedding = train_embedding[i*batch_size : (i+1)*batch_size]
            #batch_train_labels = train_labels[i*batch_size : (i+1)*batch_size].long()

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
    
    prediction = model.predict(test_embedding)
    test_acc = accuracy(prediction, test_labels)
    f1_score_weighted = f1_score(y_true=test_labels, y_pred=prediction, average='weighted')
    f1_score_micro = f1_score(y_true=test_labels, y_pred=prediction, average='micro')

    return graph_model, valid_acc, test_acc, f1_score_weighted, f1_score_micro


if __name__ == "__main__":
    if os.path.exists(HYPERPARAM_CONFIG):
        with open(HYPERPARAM_CONFIG, 'r') as f:
            hyperparam = json.loads(f.read())
    else:
        raise FileNotFoundError("hyperparam configuration file not found")

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

    trial = []
    threshold = []
    func = []
    valid_accuracy = []
    test_accuracy = []
    f1 = []
    f1_micro_score = []

    trial_run = 0
    for similarity_threshold in hyperparam["similarity_threshold"]:
        for similarity_function in hyperparam["similarity_function"]:
            print(f'Trial {trial_run} in progress ...')
            graph_model, valid_acc, test_acc, f1_weighted, f1_micro = train_and_test(graph=graph, features=features, train_labels=train_labels,
                                                                                       test_labels=test_labels, valid_labels=valid_labels,
                                                                                       emb_h_size=hyperparam["emb_h_size"],
                                                                                       mlp_h1_size=hyperparam["mlp_h1_size"],
                                                                                       mlp_h2_size=hyperparam["mlp_h2_size"], 
                                                                                       num_classes=hyperparam["num_classes"], 
                                                                                       loss_str=hyperparam["loss"],
                                                                                       learning_rate=hyperparam["learning_rate"], 
                                                                                       batch_size=hyperparam["batch_size"],
                                                                                       similarity_threshold=similarity_threshold,
                                                                                       similarity_function=similarity_function, 
                                                                                       gamma=hyperparam["gamma"])
            trial.append(trial_run)
            threshold.append(similarity_threshold)
            func.append(similarity_function)
            valid_accuracy.append(valid_acc.item())
            test_accuracy.append(test_acc.item())
            f1.append(f1_weighted)
            f1_micro_score.append(f1_micro)
            trial_run += 1

    result = {"Trial":trial, "Similarity_Threshold":threshold,
              "Similarity_Function":func, "Validation_Accuracy":valid_accuracy,
              "Testing Accuracy":test_accuracy, "F1_weighted_score":f1,
              "F1_micro_score":f1_micro_score}
    df_result = pd.DataFrame.from_dict(result)
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)

    df_result.to_csv(os.path.join(RESULT_PATH, "result_"+curr_time+".csv"))