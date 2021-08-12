import numpy as np

def simulate_data(number_of_subjects):
    mean, std = np.random.rand(), np.random.rand()
    simulated_data=list()
    for i in range(0, number_of_subjects):

        #Create matrices with 35 ROIs
        t0 = np.abs(np.random.normal(mean, std, (35,35))) % 1.0
        mean_s = mean + np.random.rand() % 0.1
        std_s = std + np.random.rand() % 0.1
        t1 = np.abs(np.random.normal(mean_s, std_s, (35,35))) % 1.0
        mean_s = mean + np.random.rand() % 0.1
        std_s = std + np.random.rand() % 0.1
        t2 = np.abs(np.random.normal(mean_s, std_s, (35,35))) % 1.0
        #Get Symmetricality
        t0 = np.maximum(t0, t0.transpose())
        t1 = np.maximum(t1, t1.transpose())
        t2 = np.maximum(t2, t2.transpose())

        #Clean The Diagonals
        t0[np.diag_indices_from(t0)] = 0
        t1[np.diag_indices_from(t1)] = 0
        t2[np.diag_indices_from(t2)] = 0

        # Save The Simulated Subjects
        """
        s = "Subject_"
        if i < 10:
            s += "0"
        s += "00" + str(i)
        t0_s = s + "_t0.txt"
        t1_s = s + "_t1.txt"
        t2_s = s + "_t2.txt"
        path='./simulated_data/'
        np.savetxt(path+t0_s, t0)
        np.savetxt(path+t1_s, t1)
        np.savetxt(path+t2_s, t2)
        """
        tsr = list()
        tsr.append(t0)
        tsr.append(t1)
        tsr.append(t2)
        tsr_array = np.asarray(tsr)
        data_path = "./simulated_data"
        file_name = data_path + "\subject_" + str(i) + "_tensor.npy"
        np.save(file_name, tsr_array)
        simulated_data.append(tsr_array)
        simulated_data_array=np.asarray(simulated_data)
    return simulated_data_array