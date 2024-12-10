# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:28:59 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:45:14 2024

@author: User
"""
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


        
        

from all_estimators import * 
from MI_hybrid_generators import * 
import matplotlib.pyplot as plt


random.seed(42)



num_samples = 200
file_to_test = 1 
num_samples_list = [100,200,300,400,500,600,700,800]


        

# save_name = 'MI_Estimation_File'+ str(file_to_test)+ '_NumSamples_200_results.pkl'
# file_dir = os.path.join(dir_path, save_name)   

# with open(file_dir, 'rb') as f:  # Python 3: open(..., 'wb')
#         [estimated_mi_mine, estimated_mi_ksg, estimated_mi_mvig, estimated_mi_VI,MI_list] = pickle.load(f)
        
# print('KSG:',np.corrcoef(np.array(estimated_mi_ksg),np.array(MI_list)))
# print('MINE:',np.corrcoef(np.array(estimated_mi_mine),np.array(MI_list)))
# print('VI:',np.corrcoef(np.array(estimated_mi_VI),np.array(MI_list)))
# print('M-VIG:',np.corrcoef(np.array(estimated_mi_mvig),np.array(MI_list)))


# print('KSG:',np.mean((np.array(estimated_mi_ksg) -np.array(MI_list))**2))
# print('MINE:',np.mean((np.array(estimated_mi_mine) -np.array(MI_list))**2))
# print('VI:',np.mean((np.array(estimated_mi_VI) -np.array(MI_list))**2))
# print('M-VIG:',np.mean((np.array(estimated_mi_mvig) -np.array(MI_list))**2))

# a = input('')

# -----------------------------------

total_epochs = 100
batch_size = 400
hidden_layer = 50
mine_est = MI_Estimator([total_epochs,batch_size,hidden_layer])
# -----------------------------------


k=2
KSG_est = MI_Estimator([k])
KSG_local_est = MI_Estimator([k])
# -----------------------------------


hidden_ratio = np.linspace(0.1,2.0,num=10)
# hidden_ratio = np.arange(1,20)/50.0
batch_size = 200
MVIG_est = MI_Estimator([hidden_ratio,batch_size])
VI_est = MI_Estimator([hidden_ratio[-1],batch_size])
# -----------------------------------

# indexes = [41]
indexes = [3, 38, 29, 59, 89, 67, 5]


file_list = [] 
estimated_mi_mine = [[] for x in range(len(indexes))] 
estimated_mi_ksg = [[] for x in range(len(indexes))]  
estimated_mi_ksg_local = [[] for x in range(len(indexes))]  
estimated_mi_mvig = [[] for x in range(len(indexes))] 
estimated_mi_VI = [[] for x in range(len(indexes))] 

file_list.append('Random_MI_Est_Experiments_num_classes_2_Dim_3_radial_var_0.002.pkl')
file_list.append('MI_Est_Experiments_num_classes_2_Dim_3_radial_var_0.03.pkl')
file_list.append('MI_Est_Experiments_num_classes_2_Dim_10_radial_var_0.005.pkl')
file_list.append('MI_Est_Experiments_num_classes_2_Dim_100_radial_var_0.001.pkl')
file_list.append('MI_Est_Experiments_num_classes_2_Dim_100_radial_var_0.1extreme_configs.pkl')




with open(file_list[file_to_test], 'rb') as f:  # Python 3: open(..., 'wb')
    [means_list,covs_list,MI_list] = pickle.load(f)




for j in range(len(indexes)):
    prob_gen = Hybrid_Generator(means_list[indexes[j]],covs_list[indexes[j]])
    
    
    for temp in range(len(num_samples_list)):
        vi = []
        mvig = [] 
        for repeat_trials in range(5):
            x_samples, y_samples = prob_gen.generate_samples(num_samples_list[temp])
            x_samples = np.array(x_samples)
            y_samples = np.array(y_samples)
            vi.append(VI_est.VInfo(x_samples, y_samples))
            mvig.append(MVIG_est.M_VIG(x_samples, y_samples))
        
        
        estimated_mi_VI[j].append(np.mean(np.array(vi)))
        # estimated_mi_mine.append(mine_est.MINE_MI(x_samples,y_samples))
        # estimated_mi_ksg.append(KSG_est.KSG(x_samples,y_samples))
        estimated_mi_mvig[j].append(np.mean(np.array(mvig)))
    
        
        print('MI:',MI_list[indexes[j]])
        # print('Estimated MI (MINE):',estimated_mi_mine[-1])
        # print('Estimated MI (KSG):',estimated_mi_ksg[-1])
        print('Estimated MI (M-VIG):',estimated_mi_mvig[j][-1])
        print('Estimated MI (V-I):',estimated_mi_VI[j][-1])
        plt.plot(estimated_mi_VI[j])
        plt.plot(estimated_mi_mvig[j])
        plt.plot(MI_list[j])
        plt.show()
    
        plt.clf() 
    

save_name = 'MI_Convergence_File'+ str(file_to_test)+ '_NumSamples_'+str(num_samples)+ '_results.pkl'
file_dir = os.path.join(dir_path, save_name)         


with open(file_dir, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([
        estimated_mi_mvig,
        estimated_mi_VI, indexes, MI_list ], f)
        
        
  
        

    
        
        

