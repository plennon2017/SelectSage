import os
import pickle
from dgl import save_graphs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import dgl
from dgl.data import DGLDataset
import torch
import lightgbm as lgb
from lightgbm import plot_importance 
from glob import glob
import matplotlib.pyplot as plt
SCALER_PATH = "./scaler"
GRAPHS_PATH = "../graphs"

node_feats = None
edge_feats = None
src_nodes_num = None
dst_nodes_num = None
binary_labels = None
multiclass_labels_int = None
train_mask_tensor = None
test_mask_tensor = None
valid_mask_tensor = None
nodes_list = None
node_feats_exploit = None
edge_feats_exploit = None
node_feats_generic = None
edge_feats_generic = None
node_feats_binary = None
edge_feats_binary = None
node_feats_multi = None
edge_feats_multi = None


class NIDS_all_features(DGLDataset):
    def __init__(self):
        super().__init__(name='NIDS_all_features')

    def process(self):
        
        node_features = torch.tensor(node_feats, dtype=torch.float32)
        edge_features = torch.tensor(edge_feats, dtype=torch.float32)
        edges_src = torch.tensor(src_nodes_num)
        edges_dst = torch.tensor(dst_nodes_num)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=len(nodes_list))
        self.graph.ndata['feat'] = node_features
        self.graph.edata['weight'] = edge_features
        self.graph.edata['label'] = torch.tensor(binary_labels, dtype=torch.int32)
        self.graph.edata['multiclass'] = torch.tensor(multiclass_labels_int, dtype=torch.int32)
        
        self.graph.edata['train_mask'] = train_mask_tensor
        self.graph.edata['test_mask'] = test_mask_tensor
        self.graph.edata['valid_mask'] = valid_mask_tensor
            

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

class NIDS_exploit_features(DGLDataset):
    def __init__(self):
        super().__init__(name='NIDS_exploit_features')

    def process(self):
        
        node_features = torch.tensor(node_feats_exploit, dtype=torch.float32)
        edge_features = torch.tensor(edge_feats_exploit, dtype=torch.float32)
        edges_src = torch.tensor(src_nodes_num)
        edges_dst = torch.tensor(dst_nodes_num)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=len(nodes_list))
        self.graph.ndata['feat'] = node_features
        self.graph.edata['weight'] = edge_features
        self.graph.edata['label'] = torch.tensor(binary_labels, dtype=torch.int32)
        self.graph.edata['multiclass'] = torch.tensor(multiclass_labels_int, dtype=torch.int32)
        
        self.graph.edata['train_mask'] = train_mask_tensor
        self.graph.edata['test_mask'] = test_mask_tensor
        self.graph.edata['valid_mask'] = valid_mask_tensor
            

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

class NIDS_generic_features(DGLDataset):
    def __init__(self):
        super().__init__(name='NIDS_generic_features')

    def process(self):
        
        node_features = torch.tensor(node_feats_generic, dtype=torch.float32)
        edge_features = torch.tensor(edge_feats_generic, dtype=torch.float32)
        edges_src = torch.tensor(src_nodes_num)
        edges_dst = torch.tensor(dst_nodes_num)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=len(nodes_list))
        self.graph.ndata['feat'] = node_features
        self.graph.edata['weight'] = edge_features
        self.graph.edata['label'] = torch.tensor(binary_labels, dtype=torch.int32)
        self.graph.edata['multiclass'] = torch.tensor(multiclass_labels_int, dtype=torch.int32)
        
        self.graph.edata['train_mask'] = train_mask_tensor
        self.graph.edata['test_mask'] = test_mask_tensor
        self.graph.edata['valid_mask'] = valid_mask_tensor
            

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
        
class NIDS_binary_features(DGLDataset):
    def __init__(self):
        super().__init__(name='NIDS_binary_features')

    def process(self):
        
        node_features = torch.tensor(node_feats_binary, dtype=torch.float32)
        edge_features = torch.tensor(edge_feats_binary, dtype=torch.float32)
        edges_src = torch.tensor(src_nodes_num)
        edges_dst = torch.tensor(dst_nodes_num)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=len(nodes_list))
        self.graph.ndata['feat'] = node_features
        self.graph.edata['weight'] = edge_features
        self.graph.edata['label'] = torch.tensor(binary_labels, dtype=torch.int32)
        self.graph.edata['multiclass'] = torch.tensor(multiclass_labels_int, dtype=torch.int32)
        
        self.graph.edata['train_mask'] = train_mask_tensor
        self.graph.edata['test_mask'] = test_mask_tensor
        self.graph.edata['valid_mask'] = valid_mask_tensor
            

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
        
class NIDS_multi_features(DGLDataset):
    def __init__(self):
        super().__init__(name='NIDS_multi_features')

    def process(self):
        
        node_features = torch.tensor(node_feats_multi, dtype=torch.float32)
        edge_features = torch.tensor(edge_feats_multi, dtype=torch.float32)
        edges_src = torch.tensor(src_nodes_num)
        edges_dst = torch.tensor(dst_nodes_num)

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=len(nodes_list))
        self.graph.ndata['feat'] = node_features
        self.graph.edata['weight'] = edge_features
        self.graph.edata['label'] = torch.tensor(binary_labels, dtype=torch.int32)
        self.graph.edata['multiclass'] = torch.tensor(multiclass_labels_int, dtype=torch.int32)
        
        self.graph.edata['train_mask'] = train_mask_tensor
        self.graph.edata['test_mask'] = test_mask_tensor
        self.graph.edata['valid_mask'] = valid_mask_tensor
            

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
        
def pipeline_preprocess(data_path):
    # Setting seed
    np.random.seed(1)

    global node_feats, edge_feats, src_nodes_num, dst_nodes_num, binary_labels, \
    multiclass_labels_int, train_mask_tensor, test_mask_tensor, valid_mask_tensor, \
    nodes_list, node_feats_exploit, edge_feats_exploit, node_feats_generic, edge_feats_generic, node_feats_binary, edge_feats_binary, node_feats_multi, edge_feats_multi

    all_files = glob(os.path.join(data_path, "*.csv"))
    feature_file = []
    data_files = []
    for each_file in all_files:
        if "features" in each_file:
            feature_file.append(each_file)
        else:
            data_files.append(each_file)

    # The provided data path should contain one csv file named features including information
    # about all the feature columns of data and the rest of the csv files should be data files 
    # containing training data

    assert len(feature_file) == 1, "More than one csv file found with name feature"
    assert len(data_files) >= 1, "No data file found in the provided path"

    cols = pd.read_csv(feature_file[0], encoding='cp1252')
    columns_list = list(cols.Name)

    # Combining all dataset files in one dataframe
    print("Loading dataset files in progress")
    df = pd.DataFrame(columns=columns_list)
    for each_file in data_files:
        print(f'Loading file {each_file}')
        temp = pd.read_csv(each_file, names=columns_list)
        print(f'{each_file} has {len(temp)} samples')
        df = pd.merge(df, temp, how='outer')
    print("Loading of dataset completed successfully")
    print(f'Total number of samples: {len(df)}')

    # ct_ftp_cmd should be numberic but it is appearing as string in dataframe. We will convert it to numeric
    df.ct_ftp_cmd = pd.to_numeric(df.ct_ftp_cmd, errors='coerce')

    # Replacing similar redundantd multiclass labels occured due to spaces
    df = df.replace([' Reconnaissance ', ' Shellcode ', ' Fuzzers ', ' Fuzzers', 'Backdoors'], 
                    ['Reconnaissance', 'Shellcode', 'Fuzzers', 'Fuzzers',  'Backdoor' ])

    # Adding a no_attack label by replacing nan
    df['attack_cat'] = df['attack_cat'].fillna('no_attack')

    # labels for multi-class classification
    multi_classes = df.attack_cat.unique()
    print(f'Multi Class\n{multi_classes}')

    # labels for binary class classification
    binary_classes = df.Label.unique()
    print(f'Binary Class\n{binary_classes}')

    # Concatenating the IP addresses and ports to make source and destination labels for the nodes
    df['SOURCE'] = df['srcip'] + ':' + df['sport'].astype(str)
    df['DESTINATION'] = df['dstip'] + ':' + df['dsport'].astype(str)

    # Drop the individual columns of IP addresses, ports Stime and Ltime
    # Also drop columns ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd becuase they have so many null values 
    #columns_to_drop = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'ct_flw_http_mthd', 
    #                   'is_ftp_login', 'ct_ftp_cmd']
                
    columns_to_drop = ['srcip', 'sport', 'dstip', 'dsport']

    df = df.drop(columns_to_drop, axis=1)

    # Rearrange the columns
    new_columns_list = list(['SOURCE', 'DESTINATION']) + list(set(columns_list) - set(columns_to_drop + ['SOURCE', 'DESTINATION']))
    df = df.reindex(columns=new_columns_list)

    # Extracting source and destination nodes along with binary and multiclass labels

    source_nodes = df['SOURCE']
    dest_nodes = df['DESTINATION']
    binary_labels = list(df['Label'])
    multiclass_labels = list(df['attack_cat'])

    # Label encoding the multiclass labels
    label_encoding = LabelEncoder()
    label_encoding.fit(multiclass_labels)
    multiclass_labels_int = label_encoding.transform(multiclass_labels)

    if not os.path.exists(SCALER_PATH):
        os.mkdir(SCALER_PATH)

    # Saving label encoding for future use
    with open(os.path.join(SCALER_PATH, 'label_encoder.pkl'), 'wb') as newfile:
        pickle.dump(label_encoding, newfile)

    # Scaling continuous data columns
    df = df.drop(['SOURCE', 'DESTINATION', 'Label', 'attack_cat'], axis=1)
    cont_columns = ['dur','sbytes','dbytes','sttl','dttl','sloss','dloss','Sload','Dload','Spkts','Dpkts',
                    'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','Sjit',
                    'Djit','Sintpkt','Dintpkt','tcprtt','synack','ackdat','ct_state_ttl','ct_srv_src',
                    'ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
                    'ct_dst_src_ltm', 'Stime', 'Ltime', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd']

    cat_columns = ['proto','state','service','is_sm_ips_ports']

    # Scaling of continuous columns
    cont_std = StandardScaler()
    cont_scaled_features = cont_std.fit_transform(df[cont_columns])

    # Onehotencoding the categorical columns
    cat_std = OneHotEncoder(sparse=False, categories='auto')
    cat_scaled_features = cat_std.fit_transform(df[cat_columns])

    # Concatenating continuous and categorical scaled features
    scaled_features = np.hstack((cont_scaled_features, cat_scaled_features))

    # list of feature names in categorical scaled features
    cat_std_categories = cat_std.categories_

    cat_onehot_cols = []
    for i, feat in zip(range(len(cat_std_categories)), cat_columns):
        for j in range(len(cat_std_categories[i])):
            cat_onehot_cols.append(feat + "-" + str(cat_std_categories[i][j]))

    # list of all feature in scaled_features
    scaled_feat_cols = cont_columns + cat_onehot_cols

    # Saving continuous feature scaling for future use
    with open(os.path.join(SCALER_PATH, 'cont_feature_scaling.pkl'), 'wb') as newfile:
        pickle.dump(cont_std, newfile)

    # Saving categorical feature scaling for future use
    with open(os.path.join(SCALER_PATH, 'cat_feature_scaling.pkl'), 'wb') as newfile:
        pickle.dump(cat_std, newfile)

    # Saving scaled feature columns
    with open('scaled_feature_columns.pkl', 'wb') as newfile:
        pickle.dump(scaled_feat_cols, newfile)

    # fitting lighgbm model to extract the most important features for classification
    gbm_model = lgb.LGBMRegressor()
    #gbm_model.fit(scaled_features, binary_labels)
    exploit_attack_label = [1 if attack=='Exploits' else 0 for attack in multiclass_labels]
    gbm_model.fit(scaled_features, exploit_attack_label)
    
    #Rough
    
    
    #feature_importances = (gbm_model.feature_importances_ / sum(gbm_model.feature_importances_)) * 100
    #results = pd.DataFrame({'cols':list(range(scaled_features.shape[1])),
    #                        'Features': scaled_feat_cols,
    #                        'Importances': gbm_model.feature_importances_})
    #selected_feats = results.loc[results.Importances > 0].sort_values(by=['Importances'], ascending = False)
    #results.sort_values(by='Importances', inplace=True)
    #print(selected_feats)
    
   
    # compute importances


    # end rough
    # fitting lighgbm model to extract the most important features for classification
    #gbm_model = lgb.LGBMRegressor(importance_type='split')
    #gbm_model.fit(scaled_features, binary_labels)
    #exploit_attack_label = [1 if attack=='Exploits' else 0 for attack in multiclass_labels]
    #gbm_model.fit(scaled_features, exploit_attack_label)

    important_feats = pd.DataFrame({'cols':list(range(scaled_features.shape[1])),
                                    'col_names':scaled_feat_cols,
                                    'feat_imp':gbm_model.feature_importances_})
    selected_feats = important_feats.loc[important_feats.feat_imp > 0].sort_values(by=['feat_imp'], ascending = False)
    print(selected_feats)
    # plotting feature importance
    
    
    
    print(f'Number of exploited features out of all features: {len(selected_feats)}/{scaled_features.shape[1]}')

    # Copying the selected features from the scaled features for all nodes
    exploited_features = scaled_features[:,list(selected_feats.cols)]

    # Saving scaled feature columns
    with open('exploited_feat_columns.pkl', 'wb') as newfile:
        pickle.dump(list(selected_feats.col_names), newfile)


#For Generic

    # fitting lighgbm model to extract the most important features for classification
    gbm_model = lgb.LGBMRegressor()
    #gbm_model.fit(scaled_features, binary_labels)
    #generic_attack_label = [1 if attack=='Generic' else 0 for attack in multiclass_labels]
    #gbm_model.fit(scaled_features, generic_attack_label)


    important_feats = pd.DataFrame({'cols':list(range(scaled_features.shape[1])),
                                    'col_names':scaled_feat_cols,
                                    'feat_imp':gbm_model.feature_importances_})
    selected_feats = important_feats.loc[important_feats.feat_imp > 0].sort_values(by=['feat_imp'], ascending = False)
    print(selected_feats)
    # plotting feature importance
    
    
    
    print(f'Number of generic features out of all features: {len(selected_feats)}/{scaled_features.shape[1]}')

    # Copying the selected features from the scaled features for all nodes
    generic_features = scaled_features[:,list(selected_feats.cols)]

    # Saving scaled feature columns
    with open('generic_feat_columns.pkl', 'wb') as newfile:
        pickle.dump(list(selected_feats.col_names), newfile)



# End generic
#For Binary


    # fitting lighgbm model to extract the most important features for classification
    gbm_model = lgb.LGBMRegressor()
    gbm_model.fit(scaled_features, binary_labels)
    


    important_feats = pd.DataFrame({'cols':list(range(scaled_features.shape[1])),
                                    'col_names':scaled_feat_cols,
                                    'feat_imp':gbm_model.feature_importances_})
    selected_feats = important_feats.loc[important_feats.feat_imp > 0].sort_values(by=['feat_imp'], ascending = False)
    print(selected_feats)
    # plotting feature importance
    
    
    
    print(f'Number of binary features out of all features: {len(selected_feats)}/{scaled_features.shape[1]}')

    # Copying the selected features from the scaled features for all nodes
    binary_features = scaled_features[:,list(selected_feats.cols)]

    # Saving scaled feature columns
    with open('binary_feat_columns.pkl', 'wb') as newfile:
        pickle.dump(list(selected_feats.col_names), newfile)



# End Binary

#For Multi


    # fitting lighgbm model to extract the most important features for classification
    gbm_model = lgb.LGBMRegressor()
    gbm_model.fit(scaled_features, multiclass_labels_int)
    


    important_feats = pd.DataFrame({'cols':list(range(scaled_features.shape[1])),
                                    'col_names':scaled_feat_cols,
                                    'feat_imp':gbm_model.feature_importances_})
    selected_feats = important_feats.loc[important_feats.feat_imp > 0].sort_values(by=['feat_imp'], ascending = False)
    print(selected_feats)
    # plotting feature importance
    
    
    
    print(f'Number of multi features out of all features: {len(selected_feats)}/{scaled_features.shape[1]}')

    # Copying the selected features from the scaled features for all nodes
    multi_features = scaled_features[:,list(selected_feats.cols)]

    # Saving scaled feature columns
    with open('multi_feat_columns.pkl', 'wb') as newfile:
        pickle.dump(list(selected_feats.col_names), newfile)



# End Multi

    # Unique nodes list for graph
    source_unique = list(pd.unique(source_nodes))
    dest_unique = list(pd.unique(dest_nodes))
    nodes_list = list(set(source_unique + dest_unique))
    src_nodes = list(source_nodes)
    dst_nodes = list(dest_nodes)

    # Conversion dictioinary from node str label to numbers
    nodes_dict = dict(zip(nodes_list, range(len(nodes_list))))

    # Conversion of source and destination str labels to numbers
    src_nodes_num = [nodes_dict[key] for key in src_nodes]
    dst_nodes_num = [nodes_dict[key] for key in dst_nodes]

    # Edge features with all features for graph
    edge_feats = scaled_features
    node_feats = np.ones((len(nodes_list), scaled_features.shape[1]), np.float32)

    # Edge features with exploited features for graph
    edge_feats_exploit = exploited_features
    node_feats_exploit = np.ones((len(nodes_list), exploited_features.shape[1]), np.float32)
    
    # Edge features with generic features for graph
    edge_feats_generic = generic_features
    node_feats_generic = np.ones((len(nodes_list), generic_features.shape[1]), np.float32)
    
    # Edge features with binary features for graph
    edge_feats_binary = binary_features
    node_feats_binary = np.ones((len(nodes_list), binary_features.shape[1]), np.float32)
    
    # Edge features with multi features for graph
    edge_feats_multi = multi_features
    node_feats_multi = np.ones((len(nodes_list), multi_features.shape[1]), np.float32)


    train_mask = []
    test_mask = []
    valid_mask = []
    for i in range(scaled_features.shape[0]):
        n = np.random.rand()
        if  n <= 0.70:
            train_mask.append(True)
            test_mask.append(False)
            valid_mask.append(False)
        elif 0.70 < n <= 0.85:
            train_mask.append(False)
            test_mask.append(True)
            valid_mask.append(False)
        else:
            train_mask.append(False)
            test_mask.append(False)
            valid_mask.append(True)

    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)
    test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool)
    valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.bool)

if __name__ == "__main__":
    pipeline_preprocess(data_path="../NB15")

    if not os.path.exists(GRAPHS_PATH):
        os.mkdir(GRAPHS_PATH)

    # Initializing all features graph
    dataset = NIDS_all_features()
    graph = dataset[0]
    print("All features graph information")
    print(graph)
    print("Saving all features graph ...")
    save_graphs(os.path.join(GRAPHS_PATH, 'graph_all_features.bin'), [graph], {})
    print("Successfully saved all features graph")

    # Initializing exploit features graph
    dataset = NIDS_exploit_features()
    graph = dataset[0]
    print("Exploit features graph information")
    print(graph)
    print("Saving exploit features graph ...")
    save_graphs(os.path.join(GRAPHS_PATH, 'graph_exploit_features.bin'), [graph], {})
    print("Successfully saved exploit features graph")
    
    # Initializing generic features graph
    dataset = NIDS_generic_features()
    graph = dataset[0]
    print("Generic features graph information")
    print(graph)
    print("Saving generic features graph ...")
    save_graphs(os.path.join(GRAPHS_PATH, 'graph_generic_features.bin'), [graph], {})
    print("Successfully saved generic features graph")
    
    # Initializing binary features graph
    dataset = NIDS_binary_features()
    graph = dataset[0]
    print("Binary features graph information")
    print(graph)
    print("Saving binary features graph ...")
    save_graphs(os.path.join(GRAPHS_PATH, 'graph_binary_features.bin'), [graph], {})
    print("Successfully saved binary features graph")
    
     # Initializing multi features graph
    dataset = NIDS_multi_features()
    graph = dataset[0]
    print("Multi features graph information")
    print(graph)
    print("Saving multi features graph ...")
    save_graphs(os.path.join(GRAPHS_PATH, 'graph_multi_features.bin'), [graph], {})
    print("Successfully saved multi features graph")


