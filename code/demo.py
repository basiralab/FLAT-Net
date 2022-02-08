"""Main function of Few-shot LeArning Training Net-work (FLAT-Net) framework
   for longitudinal brain graph evolution prediction from a few training representative templates

    ---------------------------------------------------------------------

    This file contains the implementation of the preprocessing, training and testing process of our FLAT-Net model.
        preprocessing(train_subjects,cluster_number,fold_num_sDGN,epoch_sDGN)
                Inputs:
                        train_subjects:      (n × v x l x l) tensor stacking  connectivity matrices of all training subjects
                                        n: the total number of subjects
                                        v: total number of timepoints
                                        l: the dimensions of the connectivity matrices
                        cluster_number:      predecided cluster_number for using as k in K-Means clustering algorithm.
                        fold_num_sDGN: total number of folds for cross-validation of sDGN.
                        epoch_sDGN:    total number of epochs for implementation of sDGN
                        see train method above for model and args.
                Outputs:
                        for each generated cluster, cs-CBTs for each fold
        train(cs_CBTs,fold_num_sDGN,epoch_evo,fold_num_evo,cluster_number)
                Inputs:
                        cs-CBTs:      (c x f x v) cluster-specific CBT array produced by the preprocessing phase.
                                        c:  total cluster number
                                        f:  total fold number for sDGN
                                        v:  total number of timepoints
                        fold_num_sDGN:  total number of folds for cross-validation of sDGN.
                        epoch_evo: total number of epochs for implementation of EvoGraphNet.
                        fold_num_evo: total number of folds for cross-validation of EvoGraphNet.
                        cluster_number:  predecided cluster_number for using as k in K-Means clustering algorithm.
                Output:
                        weights of the trained generators: generator1 and generator 2

        main_test(test_data,cs_CBTs,cluster_number)
                Inputs:
                        test_data:       (n × v x l x l) tensor stacking  connectivity matrices of all test subjects
                                         n: the total number of subjects
                                         v: total number of timepoints
                                         l: the dimensions of the connectivity matrices
                         cs-CBTs:      (c x f x v) cluster-specific CBT array produced by the preprocessing phase.
                                        c:  total cluster number
                                        f:  total fold number for sDGN
                                        v:  total number of timepoints
                         cluster_number:  predecided cluster_number for using as k in K-Means clustering algorithm.
                Outputs:
                        mean MAE, mean eigenvector centrality, and mean node-strength values for each fold
    To evaluate our framework we used 3-fold cross-validation strategy.
    ---------------------------------------------------------------------
    Copyright 2021 Megi Guris Özen, Istanbul Technical University.
    All rights reserved.
    """


import numpy as np
from sklearn.model_selection import KFold
#from preprocessing import *
#from model import *
#from train import *
import argparse
from simulate_data import simulate_data
from preprocessing import *
from train import *
from test import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FLAT-Net')
    parser.add_argument('--subject_number', type=int, default=58,
                        help='Subject numbers to be generated')
    parser.add_argument('--cluster_num', type=int, default=3, help='Decide Cluster Numbers')
    parser.add_argument('--folds_evo', type=int, default=3, help='How many folds for EvoGraphNet')
    parser.add_argument('--folds_sDGN', type=int, default=3, help='How many folds for sDGN')
    parser.add_argument('--folds_main', type=int, default=3, help='How many folds for main')
    parser.add_argument('--epochs_evo', type=int, default=300, help='How many epochs for EvoGraphNet')
    parser.add_argument('--epochs_sDGN', type=int, default=300, help='How many epochs for sDGN')
    args = parser.parse_args()

    fold_num_start = args.folds_main
    fold_num_evo = args.folds_evo
    fold_num_sDGN = args.folds_sDGN
    cluster_number = args.cluster_num
    number_of_subjects = args.subject_number
    epoch_evo = args.epochs_evo
    epoch_sDGN = args.epochs_sDGN

#Simulate 35x35 ROI data with the views from timepoints t0, t1, and t2, the data is represented as tensors which includes t0, t1, and t2 views
simulated_data=simulate_data(number_of_subjects)
cv = KFold(n_splits=fold_num_start, shuffle=False)
#For reporting
final_mae=list()
final_mae2=list()
final_eigen=list()
final_eigen2=list()
final_node_str=list()
final_node_str2=list()

for train_index, test_index in cv.split(simulated_data):
    train_subjects, test_subjects = simulated_data[train_index], simulated_data[test_index]
    #obtain cs-CBTs for each generated cluster
    cs_CBTs=preprocessing(train_subjects,cluster_number,fold_num_sDGN,epoch_sDGN)
    #train k sub-models with k clusters
    train(cs_CBTs,fold_num_sDGN,epoch_evo,fold_num_evo,cluster_number)
    #choose best sub-model with regard to distances from cs-CBTs for each test data
    mae,mae2,eigen,eigen2,node_str,node_str2=main_test(test_subjects,cs_CBTs,cluster_number)
    final_mae.append(mae)
    final_mae2.append(mae2)
    final_eigen.append(eigen)
    final_eigen2.append(eigen2)
    final_node_str.append(node_str)
    final_node_str2.append(node_str2)


print("Mean MAE for timepoint t1 in the experiment is: "+str(np.mean(np.asarray(final_mae))))
print("Mean MAE for timepoint t2 in the experiment is: "+str(np.mean(np.asarray(final_mae2))))
print("Mean MAE eigenvector centrality for timepoint t1 in the experiment is:"+str(np.mean(np.asarray(final_eigen))))
print("Mean MAE eigenvector centrality for timepoint t2 in the experiment is:"+str(np.mean(np.asarray(final_eigen2))))
print("Mean MAE node strength for timepoint t1 in the experiment is:"+str(np.mean(np.asarray(final_node_str))))
print("Mean MAE node strength for timepoint t2 in the experiment is:"+str(np.mean(np.asarray(final_node_str2))))
