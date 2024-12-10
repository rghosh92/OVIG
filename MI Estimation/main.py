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

from scipy import stats


random.seed(42)



num_samples = 200
file_to_test = 0



        

# save_name = 'MI_Estimation_File'+ str(file_to_test)+ '_NumSamples_1000_results.pkl'
# save_name = 'MI_Estimation_File4_NumSamples_1000_results.pkl'
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


# hidden_ratio = [np.linspace(0.1,2.0,num=10)]
hidden_ratio = np.arange(1,20)/50.0
batch_size = 200
MVIG_est = MI_Estimator([hidden_ratio,batch_size])
VI_est = MI_Estimator([hidden_ratio[-1],batch_size])
# -----------------------------------

k = 2
Mixed_est = MI_Estimator([k])
Mixed_mixed_est = MI_Estimator([k]).Mixed_Mixed_KSG
# -----------------------------------


batch_size = 400
hidden_dim = 256
layers = 2
total_epochs = 250
clip = 5.0

Infonce_est = MI_Estimator([batch_size,hidden_dim,layers,50,0.0005]).infonce_
JS_est = MI_Estimator([batch_size,hidden_dim,layers,150,0.0005]).js_
NWJ_est = MI_Estimator([batch_size,hidden_dim,layers,100,0.0005]).nwj_
SMILE_est = MI_Estimator([batch_size,hidden_dim,layers,150,0.0005,clip]).smile_

# -----------------------------------


file_list = [] 
estimated_mi_mine = [] 
estimated_mi_ksg = [] 
estimated_mi_ksg_local = [] 
estimated_mi_mixed = [] 
estimated_mi_mixed_mixed = []
estimated_mi_infonce = []
estimated_mi_js = []
estimated_mi_smile = []
estimated_mi_nwj = []  

estimated_mi_mvig = []
estimated_mi_VI = [] 

file_list.append('Random_MI_Est_Experiments_num_classes_2_Dim_3_radial_var_0.002.pkl')
file_list.append('MI_Est_Experiments_num_classes_2_Dim_3_radial_var_0.03.pkl')
file_list.append('MI_Est_Experiments_num_classes_2_Dim_10_radial_var_0.005.pkl')
file_list.append('MI_Est_Experiments_num_classes_2_Dim_100_radial_var_0.001.pkl')
file_list.append('MI_Est_Experiments_num_classes_2_Dim_100_radial_var_0.1extreme_configs.pkl')




with open(file_list[file_to_test], 'rb') as f:  # Python 3: open(..., 'wb')
    [means_list,covs_list,MI_list] = pickle.load(f)
    

for j in range(len(means_list)):
    prob_gen = Hybrid_Generator(means_list[j],covs_list[j])
    x_samples, y_samples = prob_gen.generate_samples(num_samples)
    x_samples = np.array(x_samples)
    y_samples = np.array(y_samples)
  
    # estimated_mi_VI.append(VI_est.VInfo(x_samples, y_samples))
    # estimated_mi_mine.append(mine_est.MINE_MI(x_samples,y_samples))
    # estimated_mi_ksg.append(KSG_est.KSG(x_samples,y_samples))
    # estimated_mi_mvig.append(MVIG_est.M_VIG(x_samples, y_samples))
    # estimated_mi_mixed_mixed.append(Mixed_mixed_est(x_samples, y_samples))
    # estimated_mi_infonce.append(Infonce_est(x_samples,y_samples))
    # estimated_mi_js.append(JS_est(x_samples,y_samples))
    # estimated_mi_nwj.append(NWJ_est(x_samples,y_samples))
    # estimated_mi_smile.append(SMILE_est(x_samples,y_samples))
    
    
    print('MI:',MI_list[j])
    # print('Estimated MI (MINE):',estimated_mi_mine[-1])
    # print('Estimated MI (KSG):',estimated_mi_ksg[-1])
    # print('Estimated MI (M-VIG):',estimated_mi_mvig[-1])
    # print('Estimated MI (V-I):',estimated_mi_VI[-1])
    # print('Estimated MI (JS):',estimated_mi_js[-1])
    # print('Estimated MI (SMILE):',estimated_mi_smile[-1])
    # print('Estimated MI (NWJ):',estimated_mi_nwj[-1])
    print('Estimated MI (Info_NCE):',estimated_mi_infonce[-1])
    
    

    
print('--------------RMSE------------------')
# print('MINE-JS:',np.sqrt(np.mean((np.array(estimated_mi_mine) -np.array(MI_list))**2)))

# print('JS:',np.sqrt(np.mean((np.array(estimated_mi_js) -np.array(MI_list))**2)))
print('Info_nce:',np.sqrt(np.mean((np.array(estimated_mi_infonce) -np.array(MI_list))**2)))
# print('NWJ:',np.sqrt(np.mean((np.array(estimated_mi_nwj) -np.array(MI_list))**2)))
# print('Smile:',np.sqrt(np.mean((np.array(estimated_mi_smile) -np.array(MI_list))**2)))

print('--------------MAE------------------')
# print('MINE-JS:',np.mean(np.abs(np.array(estimated_mi_mine) -np.array(MI_list))))

# print('JS:',np.mean(np.abs(np.array(estimated_mi_js) -np.array(MI_list))))
print('Info_nce:',np.mean(np.abs(np.array(estimated_mi_infonce) -np.array(MI_list))))
# print('NWJ:',np.mean(np.abs(np.array(estimated_mi_nwj) -np.array(MI_list))))
# print('Smile:',np.mean(np.abs(np.array(estimated_mi_smile) -np.array(MI_list))))


print('--------------Spearman------------------')
# print('MINE-JS:',stats.spearmanr(np.array(estimated_mi_mine),np.array(MI_list)))

# print('JS:',stats.spearmanr(np.array(estimated_mi_js),np.array(MI_list)))
print('Info_nce:',stats.spearmanr(np.array(estimated_mi_infonce),np.array(MI_list)))
# print('NWJ:',stats.spearmanr(np.array(estimated_mi_nwj),np.array(MI_list)))
# print('Smile:',stats.spearmanr(np.array(estimated_mi_smile),np.array(MI_list)))


# save_name = 'Onelayernetwork_MI_Estimation_File'+ str(file_to_test)+ '_NumSamples_'+str(num_samples)+ '_results.pkl'
# file_dir = os.path.join(dir_path, save_name)         


# with open(file_dir, 'wb') as f:  # Python 3: open(..., 'wb')
#         pickle.dump([estimated_mi_mine, 
#         estimated_mi_ksg,
#         estimated_mi_mvig,
#         estimated_mi_VI, MI_list ], f)
        
        
  
        

    
        
        

