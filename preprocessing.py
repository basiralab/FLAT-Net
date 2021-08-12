import numpy as np
from sklearn.cluster import KMeans
from sDGN import DGN


def cluster(train_data, cluster_num):
    """
    Cluster with K-Means clustering the data by considering only the t0 views.
    The cluster_num k is predefined.
    Returns an array with members from each cluster.
    """
    t0_list = list()
    clusters = [[] for i in range(cluster_num)]
    for i in range(len(train_data)):
        flatten_t0 = train_data[i][0].flatten()
        t0_list.append(flatten_t0)
    t0_array = np.asarray(t0_list)
    t0_array = t0_array.astype(np.float)
    #K-Means Clustering on t0 subject views
    cluster = KMeans(n_clusters=cluster_num, random_state=0).fit(t0_array)
    #Create an array with each clusters and their members
    for i in range(len(train_data)):
        for j in range(cluster_num):
            if cluster.labels_[i] == j:
                clusters[j].append(train_data[i])
    clusters_array = np.asarray(clusters,dtype=object)
    return clusters_array

def convert_data_to_sDGN_format(cluster):
    """
    Converting clustered data to sDGN format for generating cluster-specific CBTs.
    """
    t0 = []
    t1 = []
    t2 = []
    for i in range(len(cluster)):
        each_t0 = cluster[i][0]
        each_t0 = each_t0.reshape(35, 35, 1)
        t0.append(each_t0)
        each_t1 = cluster[i][1]
        each_t1 = each_t1.reshape(35, 35, 1)
        t1.append(each_t1)
        each_t2 = cluster[i][2]
        each_t2 = each_t2.reshape(35, 35, 1)
        t2.append(each_t2)
    else:
        t0_array = np.asarray(t0)
        t1_array = np.asarray(t1)
        t2_array = np.asarray(t2)
    return t0_array, t1_array, t2_array

def sDGN_preprocessing_operations(cluster_array,cluster_num):
    """
    Carry out data format conversion operation with each cluster.
    """
    results= [[] for i in range(cluster_num)]
    for i in range(len(results)):
        sDGN_t0,sDGN_t1,sDGN_t2=convert_data_to_sDGN_format(cluster_array[i])
        results[i].append(sDGN_t0)
        results[i].append(sDGN_t1)
        results[i].append(sDGN_t2)
    return results

def get_model_param(cluster):
    """
    Obtain necessary model parameters for sDGN.
    """
    lr = 0.0005
    CONV1 = 36
    CONV2 = 24
    CONV3 = 5
    N_Nodes = cluster.shape[1]
    N_views = cluster.shape[3]
    MODEL_PARAMS = {
        "N_ROIs": N_Nodes,
        "learning_rate": lr,
        "n_attr": cluster.shape[3],
        "Linear1": {"in": N_views, "out": CONV1},
        "conv1": {"in": 1, "out": CONV1},

        "Linear2": {"in": N_views, "out": CONV1 * CONV2},
        "conv2": {"in": CONV1, "out": CONV2},

        "Linear3": {"in": N_views, "out": CONV2 * CONV3},
        "conv3": {"in": CONV2, "out": CONV3}
    }
    return MODEL_PARAMS


def get_CBTs(cluster_results,cluster_number ,epochs, folds):
    """
    Generate cluster-specific CBTs.
    If a cluster only has one member, return this member as CBT since its the only representation.
    If a cluster has two members and the fold number is bigger, use 2 as fold number for sDGN.
    Use given fold number and cluster subjects for other scenarios.
    """
    cs_CBTs=[[] for i in range(cluster_number)]
    for i in range(cluster_number):
        t0=cluster_results[i][0].astype(np.float)
        t1 = cluster_results[i][1].astype(np.float)
        t2 = cluster_results[i][2].astype(np.float)
        if len(t0) == 1:
            print("Train Length is 1, CBT can not be produced. Original data is passed.")
            cs_CBTs[i].append(t0[0].reshape(1, 35, 35))
            cs_CBTs[i].append(t1[0].reshape(1, 35, 35))
            cs_CBTs[i].append(t2[0].reshape(1, 35, 35))
        elif len(t0) == 2:
            if folds == 2:
                _, CBTs_t0 = DGN.train_model(
                    t0,
                    model_params=get_model_param(t0),
                    n_max_epochs=epochs,
                    n_folds=folds,
                    random_sample_size=2,
                    early_stop=True,
                    model_name="DGN_TEST")
                _, CBTs_t1 = DGN.train_model(
                    t1,
                    model_params=get_model_param(t1),
                    n_max_epochs=epochs,
                    n_folds=folds,
                    random_sample_size=2,
                    early_stop=True,
                    model_name="DGN_TEST")
                _, CBTs_t2 = DGN.train_model(
                    t2,
                    model_params=get_model_param(t2),
                    n_max_epochs=epochs,
                    n_folds=folds,
                    random_sample_size=2,
                    early_stop=True,
                    model_name="DGN_TEST")
                cs_CBTs[i].append(CBTs_t0)
                cs_CBTs[i].append(CBTs_t1)
                cs_CBTs[i].append(CBTs_t2)
            elif folds > 2:
                print("Fold number is bigger than train sample number, default fold number 2 is used.")
                _, CBTs_t0 = DGN.train_model(
                    t0,
                    model_params=get_model_param(t0),
                    n_max_epochs=epochs,
                    n_folds=folds,
                    random_sample_size=2,
                    early_stop=True,
                    model_name="DGN_TEST")
                _, CBTs_t1 = DGN.train_model(
                    t1,
                    model_params=get_model_param(t1),
                    n_max_epochs=epochs,
                    n_folds=folds,
                    random_sample_size=2,
                    early_stop=True,
                    model_name="DGN_TEST")
                _, CBTs_t2 = DGN.train_model(
                    t2,
                    model_params=get_model_param(t2),
                    n_max_epochs=epochs,
                    n_folds=folds,
                    random_sample_size=2,
                    early_stop=True,
                    model_name="DGN_TEST")
                cs_CBTs[i].append(CBTs_t0)
                cs_CBTs[i].append(CBTs_t1)
                cs_CBTs[i].append(CBTs_t2)
    
        else:
            _, CBTs_t0 = DGN.train_model(
                t0,
                model_params=get_model_param(t0),
                n_max_epochs=epochs,
                n_folds=folds,
                random_sample_size=2,
                early_stop=True,
                model_name="DGN_TEST")
            _, CBTs_t1 = DGN.train_model(
                t1,
                model_params=get_model_param(t1),
                n_max_epochs=epochs,
                n_folds=folds,
                random_sample_size=2,
                early_stop=True,
                model_name="DGN_TEST")
            _, CBTs_t2 = DGN.train_model(
                t2,
                model_params=get_model_param(t2),
                n_max_epochs=epochs,
                n_folds=folds,
                random_sample_size=2,
                early_stop=True,
                model_name="DGN_TEST")
            cs_CBTs[i].append(CBTs_t0)
            cs_CBTs[i].append(CBTs_t1)
            cs_CBTs[i].append(CBTs_t2)
    return cs_CBTs

def preprocessing(train_subjects,cluster_number,fold_num_sDGN,epoch_sDGN):
    """
    implement clustering and cs-CBT generation operations.
    return cs-CBT array.
    """
    cluster_array = cluster(train_subjects, cluster_number)
    converted_cluster_data=sDGN_preprocessing_operations(cluster_array,cluster_number)
    cs_CBTs=get_CBTs(converted_cluster_data,cluster_number,epoch_sDGN,fold_num_sDGN)
    return cs_CBTs