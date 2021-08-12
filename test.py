from networkx.convert_matrix import from_numpy_array
import numpy as np
import torch
from modelEvoGraphNet import Generator
from data_utils import  create_edge_index_attribute, swap, cross_val_indices, MRDataset2
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from networkx import nx, gnm_random_graph, gnp_random_graph, erdos_renyi_graph
import networkx.algorithms.community as nx_comm
from networkx.algorithms.smallworld import random_reference
from networkx.convert_matrix import from_numpy_matrix
from sklearn.metrics import mean_absolute_error

def write_test(test_data):
    """
    This method is to write each timepoint views seperately for each test data.
    """

    data_path="./simulated_data"
    for i in range(len(test_data)):
        s0="/test"+str(i)+"_t0.txt"
        np.savetxt(data_path+s0,test_data[i][0],fmt='%s')
        s1="/test"+str(i)+"_t1.txt"
        np.savetxt(data_path+s1,test_data[i][1],fmt='%s')
        s2="/test"+str(i)+"_t2.txt"
        np.savetxt(data_path+s2,test_data[i][2],fmt='%s')

def get_model_for_cluster(this_cluster):
    """
    Get model weights for two generators for each cluster from previously trained sub-models.
    """
    device = torch.device('cuda')
    generator = Generator().to(device)
    generator2 = Generator().to(device)
    model_generator = generator
    model_generator.load_state_dict(torch.load('../weights/generator__cluster'+str(this_cluster)+'.pth'))  # generator model
    model_generator2 = generator2
    model_generator2.load_state_dict(torch.load('../weights/generator2__cluster'+str(this_cluster)+'.pth'))  # generator2 model
    return model_generator,model_generator2
def get_closes_cluster(test_data,cs_CBTs):
    """
    Choose the best sub-model for testing by computing euclidean distances between cs-CBTs and each test data.
    The appropriate sub-model indexes are return for each test data in an index list.

    """
    model_indexes=[]
    test_data=np.asarray(test_data)
    cs_CBTs=np.asarray(cs_CBTs)
    for j in range(len(test_data)):
        compare=list()
        for i in range(len(cs_CBTs)):
            x=np.linalg.norm(test_data[j][0].astype(np.float)-cs_CBTs[i][0].astype(np.float))#find distance for each test data t0 view
            compare.append(x)
        index=compare.index(min(compare))#find the minimum distance index
        model_indexes.append(index)
    return model_indexes


def get_dict_mae(dictpred, dictorg):
    dict1_list = list()
    dict2_list = list()
    for st, vals in dictpred.items():  # get predicted dict items
        dict1_list.append(vals)
    for st1, vals1 in dictorg.items():  # get original dict items
        dict2_list.append(vals1)
    dictpred_array = np.asarray(dict1_list)  # convert to np array for operations
    dictorg_array = np.asarray(dict2_list)  # convert to np array for operations
    return mean_absolute_error(dictorg_array, dictpred_array)


def eigenvector_centrality(G):
    """
    Calculate eigenvector_centrality.
    """
    max_solver_iterations = 1000000000
    return nx.eigenvector_centrality(G, weight="weight", max_iter=max_solver_iterations)


def get_node_strengths(G):
    """
        Calculate node_strength.
    """
    degrees = list()
    for i in range(35):
        degrees.append(G.degree(weight='weight')[i])

    return degrees

def main_test(test_data,cs_CBTs,cluster_number):
    """
        The main test function.
        Depending on the Euclidean distances between each test data and cs-CBTs. The best sub-model is decided and stored in an array.
        By this array for each test data the best sub-model is tested.
        From t0 views of the test data, predicted t1 and t2 views are obtained and comparison metrics are produced.
        For comparison, mean MAE, eigenvector centrality and node strength are reported and returned.
    """
    mael_list = list()
    mael2_list = list()
    eig_mae_list = list()
    eig_mae_list2 = list()
    strength_mae_list = list()
    strength_mae_list2 = list()

    device = torch.device('cuda')
    mael = torch.nn.L1Loss().to(device)
    write_test(test_data)
    sub_model_indexes=get_closes_cluster(test_data,cs_CBTs)
    test_data = MRDataset2("../data/test", "lh", subs=20)  # read the test data
    h_data_test_loader = DataLoader(test_data, batch_size=1, shuffle=True)  # create data loader
    val_step = len(h_data_test_loader)

    mae_l_val_test = 0
    mae_l_val2_test = 0
    for j, data in enumerate(h_data_test_loader):
        data = data.to(device)
        for i in range(cluster_number):
            if j == 19:
                break
            elif sub_model_indexes[j]==i:
                cluster_generator,cluster_generator2=get_model_for_cluster(i)
                fake_y = cluster_generator(data).detach()
                edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
                fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
                swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

                # data: Real source and target
                # fake_data: Real source and generated target
                mae_l_val_test += mael(cluster_generator(data), data.y).item()
                np_predicted = cluster_generator(data).cpu().detach().numpy()  # predicted as numpy array
                np.savetxt('test1predicted.txt', np_predicted)
                original_data = data.y.cpu().detach().numpy()  # test data as numpy array
                np_predicted_graph = from_numpy_matrix(np_predicted)  # convert to graph
                original_data_graph = from_numpy_matrix(original_data)  # convert to graph for operations

                eig_cent_predicted = eigenvector_centrality(np_predicted_graph) #eigenvector centrality of the predicted t1 data
                eig_cent_original = eigenvector_centrality(original_data_graph) #eigenvector centrality of the ground truth t1 data

                eig_centrality_mae = get_dict_mae(eig_cent_predicted, eig_cent_original)
                eig_mae_list.append(eig_centrality_mae)

                node_st_predicted = get_node_strengths(np_predicted_graph) #node strength of the predicted data
                node_st_original = get_node_strengths(original_data_graph) #node strength of the ground truth t1 data

                node_st_predicted_array = np.asarray(node_st_predicted)
                node_st_original_array = np.asarray(node_st_original)
                strength_mae_list.append(mean_absolute_error(node_st_original_array, node_st_predicted_array))#calculate mean absolute error for node strenght
                # Second GAN
                # Create fake data for t2 from fake data for t1
                fake_data.x = fake_data.x.detach()
                fake_y2 = cluster_generator2(fake_data)
                edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
                fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
                swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

                # fake_data: Data generated for t1
                # fake_data2: Data generated for t2 using generated data for t1
                # swapped_data2: Real t2 data
                mae_l_val2_test += mael(cluster_generator2(fake_data), data.y2).item()
                np_predicted2 = cluster_generator2(fake_data).cpu().detach().numpy()  # predicted as numpy array
                np.savetxt('test1predicted2.txt', np_predicted2)
                original_data2 = data.y2.cpu().detach().numpy()  # test data as numpy array
                np_predicted_graph2 = from_numpy_matrix(np_predicted2)  # convert to graph
                original_data_graph2 = from_numpy_matrix(original_data2)  # convert to graph for operations

                eig_cent_predicted2 = eigenvector_centrality(np_predicted_graph2)  #eigenvector centrality of the predicted data at t2
                eig_cent_original2 = eigenvector_centrality(original_data_graph2)  #eigenvector centrality of the original data at t2

                eig_centrality_mae2 = get_dict_mae(eig_cent_predicted2, eig_cent_original2)
                eig_mae_list2.append(eig_centrality_mae2)

                node_st_predicted2 = get_node_strengths(np_predicted_graph2) #node strength of the predicted data at t2
                node_st_original2 = get_node_strengths(original_data_graph2) #node strength of the original data at t2

                node_st_predicted_array2 = np.asarray(node_st_predicted2)

                node_st_original_array2 = np.asarray(node_st_original2)
                strength_mae_list2.append(mean_absolute_error(node_st_original_array2, node_st_predicted_array2))

    total_mae = mae_l_val_test / val_step
    total_mae2 = mae_l_val2_test / val_step
    mael_list.append(total_mae)
    mael2_list.append(total_mae2)

    eig_mae_array = np.asarray(eig_mae_list)
    mean_eig_mae = np.mean(eig_mae_array)

    eig_mae_array2 = np.asarray(eig_mae_list2)
    mean_eig_mae2 = np.mean(eig_mae_array2)

    strength_mae_array = np.asarray(strength_mae_list)
    mean_st_mae = np.mean(strength_mae_array)


    strength_mae_array2 = np.asarray(strength_mae_list2)
    mean_st_mae2 = np.mean(strength_mae_array2)

    print("Test is completed.")
    return mael_list,mael2_list,mean_eig_mae,mean_eig_mae2,mean_st_mae,mean_st_mae2
