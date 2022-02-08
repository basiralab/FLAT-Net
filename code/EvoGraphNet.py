import argparse
import os
import os.path as osp
import numpy as np
import math
import itertools
import copy
import pickle
from sys import exit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable
from torch.distributions import normal, kl

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool
from torch_geometric.utils import get_laplacian, to_dense_adj



from data_utils import  create_edge_index_attribute, swap, cross_val_indices, MRDataset2
from modelEvoGraphNet import Generator, Discriminator



def EvoGraphNet(epoch_number,fold_num,this_cluster,cluster_num):
    torch.manual_seed(0)  # To get the same results across experiments

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('running on GPU')
    else:
        device = torch.device("cpu")
        print('running on CPU')
    
    h_data = MRDataset2("./simulated_data", "lh", subs=cluster_num)

    # Parameters

    batch_size = 1
    lr_G = 0.01
    lr_D = 0.0002
    num_epochs = epoch_number#I have used 300 before but lets get it as a parameter.
    
    connectomes = 1
    train_generator = 1

    id_e=2
    # Coefficients for loss
    i_coeff = 2.0
    g_coeff = 2.0
    kl_coeff = 0.001
    tp_coeff = 0.0

    loss='BCE'
    tr_st="same"
    # Training
    decay=0.0
    loss_dict = {"BCE": torch.nn.BCELoss().to(device),
                "LS": torch.nn.MSELoss().to(device)}


    adversarial_loss = loss_dict[loss.upper()]
    identity_loss = torch.nn.L1Loss().to(device)  # Will be used in training
    msel = torch.nn.MSELoss().to(device)
    mael = torch.nn.L1Loss().to(device)  # Not to be used in training (Measure generator success)
    counter_g, counter_d = 0, 0
    tp = torch.nn.MSELoss().to(device) # Used for node strength
    
    train_ind, val_ind = cross_val_indices(fold_num, len(h_data))

    # Saving the losses for the future
    gen_mae_losses_tr = None
    disc_real_losses_tr = None
    disc_fake_losses_tr = None
    gen_mae_losses_val = None
    disc_real_losses_val = None
    disc_fake_losses_val = None
    gen_mae_losses_tr2 = None
    disc_real_losses_tr2 = None
    disc_fake_losses_tr2 = None
    gen_mae_losses_val2 = None
    disc_real_losses_val2 = None
    disc_fake_losses_val2 = None
    k1_train_s = None
    k2_train_s = None
    k1_val_s = None
    k2_val_s = None
    tp1_train_s = None
    tp2_train_s = None
    tp1_val_s = None
    tp2_val_s = None
    gan1_train_s = None
    gan2_train_s = None
    gan1_val_s = None
    gan2_val_s = None


    mae_list_val1=list()
    mae_list_val2=list()
    std_list_val1=list()
    std_list_val2=list()

    # Cross Validation
    for fold in range(fold_num):
        train_set, val_set = h_data[list(train_ind[fold])], h_data[list(val_ind[fold])]
        
        h_data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        h_data_test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        val_step = len(h_data_test_loader)
        

        for data in h_data_train_loader:  # Determine the maximum number of samples in a batch
            data_size = data.x.size(0)
            break

        # Create generators and discriminators
        generator = Generator().to(device)
        generator2 = Generator().to(device)
        discriminator = Discriminator().to(device)
        discriminator2 = Discriminator().to(device)

        optimizer_G = torch.optim.AdamW(generator.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=decay)
        optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=decay)
        optimizer_G2 = torch.optim.AdamW(generator2.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=decay)
        optimizer_D2 = torch.optim.AdamW(discriminator2.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=decay)

        total_step = len(h_data_train_loader)
        
        real_label = torch.ones((data_size, 1)).to(device)
        fake_label = torch.zeros((data_size, 1)).to(device)

        
        # Will be used for reporting
        real_losses, fake_losses, mse_losses, mae_losses = list(), list(), list(), list()
        real_losses_val, fake_losses_val, mse_losses_val, mae_losses_val = list(), list(), list(), list()

        real_losses2, fake_losses2, mse_losses2, mae_losses2 = list(), list(), list(), list()
        real_losses_val2, fake_losses_val2, mse_losses_val2, mae_losses_val2 = list(), list(), list(), list()

        k1_losses, k2_losses, k1_losses_val, k2_losses_val = list(), list(), list(), list()
        tp_losses_1_tr, tp_losses_1_val, tp_losses_2_tr, tp_losses_2_val = list(), list(), list(), list()
        gan_losses_1_tr, gan_losses_1_val, gan_losses_2_tr, gan_losses_2_val = list(), list(), list(), list()


        total_loss_1,total_loss_2,total_loss_1_val,total_loss_2_val=list(),list(),list(),list()


        for epoch in range(num_epochs):
            # Reporting
            r, f, d, g, mse_l, mae_l = 0, 0, 0, 0, 0, 0
            r_val, f_val, d_val, g_val, mse_l_val, mae_l_val = 0, 0, 0, 0, 0, 0
            k1_train, k2_train, k1_val, k2_val = 0.0, 0.0, 0.0, 0.0
            r2, f2, d2, g2, mse_l2, mae_l2 = 0, 0, 0, 0, 0, 0
            r_val2, f_val2, d_val2, g_val2, mse_l_val2, mae_l_val2 = 0, 0, 0, 0, 0, 0
            tp1_tr, tp1_val, tp2_tr, tp2_val = 0.0, 0.0, 0.0, 0.0
            gan1_tr, gan1_val, gan2_tr, gan2_val = 0.0, 0.0, 0.0, 0.0
        
            # Train
            generator.train()
            discriminator.train()
            generator2.train()
            discriminator2.train()
        
            for i, data in enumerate(h_data_train_loader):
                data = data.to(device)
                
                optimizer_D.zero_grad()

                # Train the discriminator
                # Create fake data
                fake_y = generator(data).detach()
                edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
                fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
                swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

                # data: Real source and target
                # fake_data: Real source and generated target
                real_loss = adversarial_loss(discriminator(swapped_data, data), real_label[:data.x.size(0), :])
                fake_loss = adversarial_loss(discriminator(fake_data, data), fake_label[:data.x.size(0), :])
                loss_D = torch.mean(real_loss + fake_loss) / 2
                r += real_loss.item()
                f += fake_loss.item()
                d += loss_D.item()

                # Depending on the chosen training method, we might update the parameters of the discriminator
                if (epoch % 2 == 1 and tr_st == "turns") or tr_st == "same" or counter_d >= id_e:
                    loss_D.backward(retain_graph=True)
                    optimizer_D.step()

                # Train the generator
                optimizer_G.zero_grad()

                # Adversarial Loss
                fake_data.x = generator(data)
                gan_loss = torch.mean(adversarial_loss(discriminator(fake_data, data), real_label[:data.x.size(0), :]))
                gan1_tr += gan_loss.item()

                # KL Loss
                kl_loss = kl.kl_divergence(normal.Normal(fake_data.x.mean(dim=1), fake_data.x.std(dim=1)),
                                        normal.Normal(data.y.mean(dim=1), data.y.std(dim=1))).sum()

                # Topology Loss
                tp_loss = tp(fake_data.x.sum(dim=-1), data.y.sum(dim=-1))
                tp1_tr += tp_loss.item()

                # Identity Loss is included in the end
                loss_G = i_coeff * identity_loss(generator(swapped_data), data.y) + g_coeff * gan_loss + kl_coeff * kl_loss + tp_coeff * tp_loss
                g += loss_G.item()
                if (epoch % 2 == 0 and tr_st == "turns") or tr_st == "same" or counter_g < id_e:
                    loss_G.backward(retain_graph=True)
                    optimizer_G.step()
                k1_train += kl_loss.item()
                mse_l += msel(generator(data), data.y).item()
                mae_l += mael(generator(data), data.y).item()

                # Training of the second part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                optimizer_D2.zero_grad()

                # Train the discriminator2

                # Create fake data for t2 from fake data for t1
                fake_data.x = fake_data.x.detach()
                fake_y2 = generator2(fake_data).detach()
                edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
                fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
                swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

                # fake_data: Data generated for t1
                # fake_data2: Data generated for t2 using generated data for t1
                # swapped_data2: Real t2 data
                real_loss = adversarial_loss(discriminator2(swapped_data2, fake_data), real_label[:data.x.size(0), :])
                fake_loss = adversarial_loss(discriminator2(fake_data2, fake_data), fake_label[:data.x.size(0), :])
                loss_D = torch.mean(real_loss + fake_loss) / 2
                r2 += real_loss.item()
                f2 += fake_loss.item()
                d2 += loss_D.item()

                if (epoch % 2 == 1 and tr_st == "turns") or tr_st == "same" or counter_d >= id_e:
                    loss_D.backward(retain_graph=True)
                    optimizer_D2.step()

                # Train generator2
                optimizer_G2.zero_grad()

                # Adversarial Loss
                fake_data2.x = generator2(fake_data)
                gan_loss = torch.mean(adversarial_loss(discriminator2(fake_data2, fake_data), real_label[:data.x.size(0), :]))
                gan2_tr += gan_loss.item()

                # Topology Loss
                tp_loss = tp(fake_data2.x.sum(dim=-1), data.y2.sum(dim=-1))
                tp2_tr += tp_loss.item()

                # KL Loss
                kl_loss = kl.kl_divergence(normal.Normal(fake_data2.x.mean(dim=1), fake_data2.x.std(dim=1)),
                                        normal.Normal(data.y2.mean(dim=1), data.y2.std(dim=1))).sum()

                # Identity Loss
                loss_G = i_coeff * identity_loss(generator(swapped_data2), data.y2) + g_coeff * gan_loss + kl_coeff * kl_loss + tp_coeff * tp_loss
                g2 += loss_G.item()
                if (epoch % 2 == 0 and tr_st == "turns") or tr_st == "same" or counter_g < id_e:
                    loss_G.backward(retain_graph=True)
                    optimizer_G2.step()
            
                k2_train += kl_loss.item()
                mse_l2 += msel(generator2(fake_data), data.y2).item()
                mae_l2 += mael(generator2(fake_data), data.y2).item()

            # Validate
            generator.eval()
            discriminator.eval()
            generator2.eval()
            discriminator2.eval()

            for i, data in enumerate(h_data_test_loader):
                data = data.to(device)
                # Train the discriminator
                # Create fake data
                
                fake_y = generator(data).detach()
                edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
                fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
                swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

                # data: Real source and target
                # fake_data: Real source and generated target
                real_loss = adversarial_loss(discriminator(swapped_data, data), real_label[:data.x.size(0), :])
                fake_loss = adversarial_loss(discriminator(fake_data, data), fake_label[:data.x.size(0), :])
                loss_D = torch.mean(real_loss + fake_loss) / 2
                r_val += real_loss.item()
                f_val += fake_loss.item()
                d_val += loss_D.item()

                # Adversarial Loss
                fake_data.x = generator(data)
                gan_loss = torch.mean(adversarial_loss(discriminator(fake_data, data), real_label[:data.x.size(0), :]))
                gan1_val += gan_loss.item()

                # Topology Loss
                tp_loss = tp(fake_data.x.sum(dim=-1), data.y.sum(dim=-1))
                tp1_val += tp_loss.item()

                kl_loss = kl.kl_divergence(normal.Normal(fake_data.x.mean(dim=1), fake_data.x.std(dim=1)),
                                        normal.Normal(data.y.mean(dim=1), data.y.std(dim=1))).sum()

                # Identity Loss

                loss_G = i_coeff * identity_loss(generator(swapped_data), data.y) + g_coeff * gan_loss * kl_coeff * kl_loss
                g_val += loss_G.item()
                mse_l_val += msel(generator(data), data.y).item()
                mae_l_val += mael(generator(data), data.y).item()
                k1_val += kl_loss.item()

                # Second GAN

                # Create fake data for t2 from fake data for t1
                fake_data.x = fake_data.x.detach()
                fake_y2 = generator2(fake_data)
                edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
                fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
                swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

                # fake_data: Data generated for t1
                # fake_data2: Data generated for t2 using generated data for t1
                # swapped_data2: Real t2 data
                real_loss = adversarial_loss(discriminator2(swapped_data2, fake_data), real_label[:data.x.size(0), :])
                fake_loss = adversarial_loss(discriminator2(fake_data2, fake_data), fake_label[:data.x.size(0), :])
                loss_D = torch.mean(real_loss + fake_loss) / 2
                r_val2 += real_loss.item()
                f_val2 += fake_loss.item()
                d_val2 += loss_D.item()

                # Adversarial Loss
                fake_data2.x = generator2(fake_data)
                gan_loss = torch.mean(adversarial_loss(discriminator2(fake_data2, fake_data), real_label[:data.x.size(0), :]))
                gan2_val += gan_loss.item()

                # Topology Loss
                tp_loss = tp(fake_data2.x.sum(dim=-1), data.y2.sum(dim=-1))
                tp2_val += tp_loss.item()

                # KL Loss
                kl_loss = kl.kl_divergence(normal.Normal(fake_data2.x.mean(dim=1), fake_data2.x.std(dim=1)),
                                        normal.Normal(data.y2.mean(dim=1), data.y2.std(dim=1))).sum()
                k2_val += kl_loss.item()

                # Identity Loss
                loss_G = i_coeff * identity_loss(generator(swapped_data2), data.y2) + g_coeff * gan_loss + kl_coeff * kl_loss
                g_val2 += loss_G.item()
                mse_l_val2 += msel(generator2(fake_data), data.y2).item()
                mae_l_val2 += mael(generator2(fake_data), data.y2).item()

            if tr_st == 'idle':
                counter_g += 1
                counter_d += 1
                if counter_g == 2 * id_e:
                    counter_g = 0
                    counter_d = 0


            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'[Train]: D Loss: {d / total_step:.5f}, G Loss: {g / total_step:.5f} R Loss: {r / total_step:.5f}, F Loss: {f / total_step:.5f}, MSE: {mse_l / total_step:.5f}, MAE: {mae_l / total_step:.5f}')
            print(f'[Val]: D Loss: {d_val / val_step:.5f}, G Loss: {g_val / val_step:.5f} R Loss: {r_val / val_step:.5f}, F Loss: {f_val / val_step:.5f}, MSE: {mse_l_val / val_step:.5f}, MAE: {mae_l_val / val_step:.5f}')
            print(f'[Train]: D2 Loss: {d2 / total_step:.5f}, G2 Loss: {g2 / total_step:.5f} R2 Loss: {r2 / total_step:.5f}, F2 Loss: {f2 / total_step:.5f}, MSE: {mse_l2 / total_step:.5f}, MAE: {mae_l2 / total_step:.5f}')
            print(f'[Val]: D2 Loss: {d_val2 / val_step:.5f}, G2 Loss: {g_val2 / val_step:.5f} R2 Loss: {r_val2 / val_step:.5f}, F2 Loss: {f_val2 / val_step:.5f}, MSE: {mse_l_val2 / val_step:.5f}, MAE: {mae_l_val2 / val_step:.5f}')

           
            mae_losses_val.append(mae_l_val / val_step) 
            mae_losses_val2.append(mae_l_val2 / val_step)
            
          
            gan_losses_1_val.append(gan1_val / val_step)
            
            gan_losses_2_val.append(gan2_val / val_step)
            
            
            
            total_loss_1_val.append(g_val/val_step)
            total_loss_2_val.append(g_val2/val_step)
        
        

        

        # Save the losses
        if gen_mae_losses_tr is None:
            gen_mae_losses_tr = mae_losses
            disc_real_losses_tr = real_losses
            disc_fake_losses_tr = fake_losses
            gen_mae_losses_val = mae_losses_val
            disc_real_losses_val = real_losses_val
            disc_fake_losses_val = fake_losses_val
            gen_mae_losses_tr2 = mae_losses2
            disc_real_losses_tr2 = real_losses2
            disc_fake_losses_tr2 = fake_losses2
            gen_mae_losses_val2 = mae_losses_val2
            disc_real_losses_val2 = real_losses_val2
            disc_fake_losses_val2 = fake_losses_val2
            k1_train_s = k1_losses
            k2_train_s = k2_losses
            k1_val_s = k1_losses_val
            k2_val_s = k2_losses_val
            tp1_train_s = tp_losses_1_tr
            tp2_train_s = tp_losses_2_tr
            tp1_val_s = tp_losses_1_val
            tp2_val_s = tp_losses_2_val
            gan1_train_s = gan_losses_1_tr
            gan2_train_s = gan_losses_2_tr
            gan1_val_s = gan_losses_1_val
            gan2_val_s = gan_losses_2_val
        else:
            gen_mae_losses_tr = np.vstack([gen_mae_losses_tr, mae_losses])
            disc_real_losses_tr = np.vstack([disc_real_losses_tr, real_losses])
            disc_fake_losses_tr = np.vstack([disc_fake_losses_tr, fake_losses])
            gen_mae_losses_val = np.vstack([gen_mae_losses_val, mae_losses_val])
            disc_real_losses_val = np.vstack([disc_real_losses_val, real_losses_val])
            disc_fake_losses_val = np.vstack([disc_fake_losses_val, fake_losses_val])
            gen_mae_losses_tr2 = np.vstack([gen_mae_losses_tr2, mae_losses2])
            disc_real_losses_tr2 = np.vstack([disc_real_losses_tr2, real_losses2])
            disc_fake_losses_tr2 = np.vstack([disc_fake_losses_tr2, fake_losses2])
            gen_mae_losses_val2 = np.vstack([gen_mae_losses_val2, mae_losses_val2])
            disc_real_losses_val2 = np.vstack([disc_real_losses_val2, real_losses_val2])
            disc_fake_losses_val2 = np.vstack([disc_fake_losses_val2, fake_losses_val2])
            k1_train_s = np.vstack([k1_train_s, k1_losses])
            k2_train_s = np.vstack([k2_train_s, k2_losses])
            k1_val_s = np.vstack([k1_val_s, k1_losses_val])
            k2_val_s = np.vstack([k2_val_s, k2_losses_val])
            tp1_train_s = np.vstack([tp1_train_s, tp_losses_1_tr])
            tp2_train_s = np.vstack([tp2_train_s, tp_losses_2_tr])
            tp1_val_s = np.vstack([tp1_val_s, tp_losses_1_val])
            tp2_val_s = np.vstack([tp2_val_s, tp_losses_2_val])
            gan1_train_s = np.vstack([gan1_train_s, gan_losses_1_tr])
            gan2_train_s = np.vstack([gan2_train_s, gan_losses_2_tr])
            gan1_val_s = np.vstack([gan1_val_s, gan_losses_1_val])
            gan2_val_s = np.vstack([gan2_val_s, gan_losses_2_val])

            # Save the models
    torch.save(generator.state_dict(), "../weights/generator_"  + "_"  + "cluster" + str(this_cluster)+".pth")
    torch.save(discriminator.state_dict(), "../weights/discriminator_"  + "_" + "cluster" + str(this_cluster)+".pth")
    torch.save(generator2.state_dict(),
            "../weights/generator2_"  + "_" +"cluster"  + str(this_cluster)+".pth")
    torch.save(discriminator2.state_dict(),
            "../weights/discriminator2_"  + "_"  + "cluster" + str(this_cluster)+".pth")

    del generator
    del discriminator

    del generator2
    del discriminator2


    print(f"Training Complete for cluster "+str(this_cluster))

