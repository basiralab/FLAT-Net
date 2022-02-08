import numpy as np
from sklearn.cluster import KMeans
from sDGN import DGN
from EvoGraphNet import EvoGraphNet

def write_CBTs_Evo(cs_CBTs,cluster_number,sdgn_folds):
    """
     write CBTs generated from folds to simulated_data
    """
    for j in range(sdgn_folds):
        t0_CBT = np.asarray(cs_CBTs[cluster_number][j][0])
        t1_CBT = np.asarray(cs_CBTs[cluster_number][j][1])
        t2_CBT = np.asarray(cs_CBTs[cluster_number][j][2])
        data_path = "./simulated_data/"
        for i in range(len(cs_CBTs[0])):
            s0 = "/CBT_" +str(j)+ "_t0.txt"
            np.savetxt(data_path + s0, t0_CBT)
            s1 = "/CBT_" +str(j)+"_t1.txt"
            np.savetxt(data_path + s1, t1_CBT)
            s2 = "/CBT_"+str(j)+ "_t2.txt"
            np.savetxt(data_path + s2, t2_CBT)

def train(cs_CBTs,fold_num_sDGN,epoch_evo,fold_num_evo,cluster_number):
    """
         for each cluster train one EvoGraphNet sub-model
    """
    for i in range(cluster_number):
        write_CBTs_Evo(cs_CBTs,i,fold_num_sDGN)
        EvoGraphNet(epoch_evo,fold_num_evo,i,cluster_number)
