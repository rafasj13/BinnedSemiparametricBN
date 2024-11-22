import pandas as pd
import numpy as np
import pybnesian as pbn
import os
import time
from utils.util_draw import * 
from utils.util_metrics import *
import glob
import json

def get_nodes_and_parents(key): #return nodes and parents
    if key == 1:
        return ['A','B','C','D','E','F','G'], 3
    if key == 2:
        return ['A','B','C','D','E','F','G','H','I','J','K','L','M'], 5
        
    elif key == 3:
        return ['A','B','C','D','E','F','G','H'], 1
    elif key == 4:
        return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'], 1
            

init, fini, iters = (14, 15, 15)
kcv, patience = (5, 3)
simu = True
ntest = 2048
jsondags = {}

# Define all controller configurations and their labels
controller_config = {
    '1.SPBN': {'controller': {}, 'key': 'SPBN', 'args': {'linear': False, 'use_fft': False}},
    '2.BSBN': {'controller': {}, 'key': 'BSBN', 'args': {'linear': False, 'use_fft': False}},
    '3.BSBN-Linear': {'controller': {}, 'key': 'BSBN-Linear', 'args': {'linear': True, 'use_fft': False}},
    '4.BSBN-FFT': {'controller': {}, 'key': 'BSBN-FFT', 'args': {'linear': False, 'use_fft': True}},
    '5.BSBN-FFT-Linear': {'controller': {}, 'key': 'BSBN-FFT-Linear', 'args': {'linear': True, 'use_fft': True}},
    '6.BSBN-FFT-SKDE': {'controller': {}, 'key': 'BSBN-FFT-SKDE', 'args': {'linear': False, 'use_fft': True}},
    '7.BSBN-FFT-SKDE-Linear': {'controller': {}, 'key': 'BSBN-FFT-SKDE-Linear', 'args': {'linear': True, 'use_fft': True}},
}
config = ([[100],[50,80],[100]], [[3,4,1],[2],[2]], [13,14,13]) # M, simu_key, power

for kexp, (grids, simulations, new_init) in enumerate(zip(*config)):
    for power in range(new_init, fini):
        for M in grids:
            n = 2**power
            results = []

            for simu_key in simulations:
                nodes, parents = get_nodes_and_parents(simu_key)
                print(f'\nSample size {n}, Simulated data {simu_key}, Grid size {M}')
                print(get_config(simu_key)['arcs'])

                # Initialize controllers dynamically
                for key in controller_config:
                    if M not in controller_config[key]['controller']:
                        controller_config[key]['controller'][M] = {}
                    controller = controller_config[key]['controller'][M]
                    controller_config[key]['controller'][M] = ExperimentsController(controller, nodes, iters)

                i = 0
                while i < iters:
                    
                    traindat, testdat, pool = controller_config['1.SPBN']['controller'][M].set_up(
                        n, ntest, simulate={'bool': simu, 'key': simu_key}, seeds=(i, 255 + i)
                    )
                    print(i, '->', traindat.shape, testdat.shape)

                    # Reference model
                    model_ref = controller_config['1.SPBN']['controller'][M].get_simulate_ref()
                    model_ref.fit(traindat)
                    start = time.time()
                    logl_ref = model_ref.logl(testdat)
                    end = time.time()
                    test_time_ref = end-start

                    # Iterate through configurations
                    for nc,(key, config) in enumerate(controller_config.items()):
                        try:
                            configcp = config.copy()
                            configcp['args']['grid'] = M
                            if "SPBN" in key:
                                vl = pbn.ValidatedLikelihood(traindat, k=kcv)
                            elif key[0] in ["4", "5"] and simu_key not in [3, 4]:
                                    config['controller'][M].append(i)
                                    continue
                            elif key[0] in ["6", "7"]  and simu_key in [3, 4]:
                                    config['controller'][M].append(i)
                                    continue
                            else:
                                vl = pbn.ValidatedLikelihoodFT(traindat, k=kcv, **configcp['args'])

                            model, train_time, test_time, logl = ExperimentsController.train_model(
                                key, traindat, testdat, pool, vl, nodes, patience, parents, **configcp['args']
                            )

                            config['controller'][M].prepare_dags(model, model_ref)
                            config['controller'][M].append(
                                i, times={'train_new': train_time, 'test_new': test_time, 'train_ref': -1, 'test_ref': test_time_ref},
                                logl={'new': logl, 'ref': logl_ref}
                            )

                            exceptbool = False
                        except Exception as e:
                            print(f'Exception in {key}: {e}')
                            for j in range(nc):
                                controlist = list(controller_config.keys())
                                controller_config[controlist[j]]['controller'][M].pop()

                            exceptbool = True
                            break

                    i = i+1 if not exceptbool else i

                # Save results dynamically
                
                for key, config in controller_config.items():   
                    controller_config[key]['controller'][M] = config['controller'][M].jsonify(n, f'simu{simu_key}', key, 'REF')
                    
                # Aggregate and save all results
                all_res = pd.concat([
                    pd.DataFrame(config['controller'][M][n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
                    for config in controller_config.values()])
                print(all_res)

                svpath = f'results/exp_simu/Mfix{kexp+3}'
                os.makedirs(svpath, exist_ok=True)

                all_results = {key: config['controller'][M] for key,config in controller_config.items()}
                with open(f'{svpath}/simu_all_{M}.json', 'w') as json_file:
                    json.dump(all_results, json_file)





















# init,fini, iters = (12,15,15)
# kcv, patience = (5, 5)
# simu = True
# ntest = 2048
# json_DAGS, json_DAGS2, json_DAGS3, json_DAGS4, json_DAGS5, json_DAGS6, json_DAGS7 = {}, {}, {}, {}, {}, {}, {}
# for power in range(init,fini):
#     for M in  [30,50,80,100]:
#         n=2**power
#         results = []
        

#         for simu_key in [1,2,3,4]:
#             nodes, parents = get_nodes_and_parents(simu_key)
#             print(f'\nSample size {n}, Simulated data {simu_key}, Grid size {M}')
#             print(get_config(simu_key)['arcs'])

#             controller2 = ExperimentsController(json_DAGS2, nodes, iters) # spbn

#             controller = ExperimentsController(json_DAGS, nodes, iters) # bsbn simple
#             controller6 = ExperimentsController(json_DAGS6, nodes, iters) # bsbn linear
            
#             controller4 = ExperimentsController(json_DAGS4, nodes, iters) # bsbn-fft-skde simple
#             controller7 = ExperimentsController(json_DAGS7, nodes, iters) # bsbn-fft-skde linear

#             controller3 = ExperimentsController(json_DAGS3, nodes, iters) # bsbn-fft simple
#             controller5 = ExperimentsController(json_DAGS5, nodes, iters) # bsbn-fft linear 


#             i=0
#             while i < iters:

#                 traindat, testdat, pool = controller.set_up(n, ntest, simulate = {'bool':simu, 'key':simu_key}, seeds=(i,255+i))
#                 print(i,'->',traindat.shape, testdat.shape)
                
#                 # ref
#                 model_ref = controller.get_simulate_ref()
#                 model_ref.fit(traindat)
#                 start = time.time()
#                 logl_ref = model_ref.logl(testdat)
#                 end = time.time()
#                 test_time_ref = end - start


#                 # spbn
#                 try:
#                     vl = pbn.ValidatedLikelihood(traindat, k=kcv)
#                     spbn, train_time_spbn, test_time_spbn, logl_spbn  = ExperimentsController.train_model('SPBN',
#                                                                                             traindat, testdat, pool, vl, nodes, patience, parents)
#                     controller2.prepare_dags(spbn, model_ref)
#                     controller2.append(i, times = {'train_new': train_time_spbn, 'test_new': test_time_spbn, 
#                                                 'train_ref': -1, 'test_ref': test_time_ref}, 
#                                         logl = {'new': logl_spbn, 'ref': logl_ref})
#                 except Exception as e:
#                     print(f'Exception in SPBN: {e}')
#                     continue
                
#                 # bsbn
#                 try:
#                     vlBin = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=False, use_fft=False)
#                     bsbn, train_time_bsbn, test_time_bsbn, logl_bsbn  = ExperimentsController.train_model('BSBN', 
#                                                                                             traindat, testdat, pool, vlBin, nodes, patience, parents,
#                                                                                             **{'linear': False, 'FFT': False, 'M': M})
#                     controller.prepare_dags(bsbn, model_ref)
#                     controller.append(i, times = {'train_new': train_time_bsbn, 'test_new': test_time_bsbn, 
#                                                 'train_ref': -1, 'test_ref': test_time_ref}, 
#                                         logl = {'new': logl_bsbn, 'ref': logl_ref})
#                 except Exception as e:
#                     print(f'Exception in BSBN: {e}')
#                     controller2.pop()
#                     continue    
                
#                 # bsbn linear
#                 try:
#                     vlBin = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=True, use_fft=False)
#                     bsbn_linear, train_time_bsbn_linear, test_time_bsbn_linear, logl_bsbn_linear  = ExperimentsController.train_model('BSBN-Linear', 
#                                                                                                     traindat, testdat, pool, vlBin, nodes, patience, parents,
#                                                                                                     **{'linear': True, 'FFT': False, 'M': M})
#                     controller6.prepare_dags(bsbn_linear, model_ref)
#                     controller6.append(i, times = {'train_new': train_time_bsbn_linear, 'test_new': test_time_bsbn_linear, 
#                                                 'train_ref': -1, 'test_ref': test_time_ref}, 
#                                         logl = {'new': logl_bsbn_linear, 'ref': logl_ref})
#                 except Exception as e:
#                     print(f'Exception in BSBN-Linear: {e}')
#                     controller2.pop()
#                     controller.pop()
#                     continue
                
#                 if simu_key in [3,4]:
#                     # bsbn-fft
#                     try:
#                         vlFT = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=False, use_fft=True)
#                         bsbn_fft, train_time_bsbn_fft, test_time_bsbn_fft, logl_bsbn_fft  = ExperimentsController.train_model('BSBN-FFT', 
#                                                                                                 traindat, testdat, pool, vlFT, nodes, patience, parents,
#                                                                                                 **{'linear': False, 'FFT': True, 'M': M})
#                         controller3.prepare_dags(bsbn_fft, model_ref)
#                         controller3.append(i, times = {'train_new': train_time_bsbn_fft, 'test_new': test_time_bsbn_fft, 
#                                                     'train_ref': -1, 'test_ref': test_time_ref}, 
#                                             logl = {'new': logl_bsbn_fft, 'ref': logl_ref})
#                     except Exception as e:
#                         print(f'Exception in BSBN-FFT: {e}')
#                         controller2.pop()
#                         controller.pop()
#                         controller6.pop()
#                         continue

#                     controller4.append(i)


#                     # bsbn-fft linear
#                     try:
#                         vlFT_linear = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=True, use_fft=True)
#                         bsbn_fft_linear, train_time_bsbn_fft_linear, test_time_bsbn_fft_linear, logl_bsbn_fft_linear  = ExperimentsController.train_model('BSBN-FFT-Linear', 
#                                                                                                                         traindat, testdat, pool, vlFT_linear, nodes, patience, parents,
#                                                                                                                         **{'linear': True, 'FFT': True, 'M': M})
#                         controller5.prepare_dags(bsbn_fft_linear, model_ref)
#                         controller5.append(i, times = {'train_new': train_time_bsbn_fft_linear, 'test_new': test_time_bsbn_fft_linear, 
#                                                     'train_ref': -1, 'test_ref': test_time_ref}, 
#                                             logl = {'new': logl_bsbn_fft_linear, 'ref': logl_ref})
#                     except Exception as e:
#                         print(f'Exception in BSBN-FFT-Linear: {e}')
#                         controller2.pop()
#                         controller.pop()
#                         controller6.pop()
#                         controller3.pop()
#                         continue
                    
#                     controller7.append(i)


#                 else:
#                     controller3.append(i)

#                     # bsbn-fft-skde
#                     try:
#                         vlFT = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=False, use_fft=True)
#                         bsbn_fft_skde, train_time_bsbn_fft_skde, test_time_bsbn_fft_skde, logl_bsbn_fft_skde  = ExperimentsController.train_model('BSBN-FFT-SKDE', 
#                                                                                                                 traindat, testdat, pool, vlFT, nodes, patience, parents,
#                                                                                                                 **{'linear': False, 'FFT': True, 'M': M})
#                         controller4.prepare_dags(bsbn_fft_skde, model_ref)
#                         controller4.append(i, times = {'train_new': train_time_bsbn_fft_skde, 'test_new': test_time_bsbn_fft_skde, 
#                                                     'train_ref': -1, 'test_ref': test_time_ref}, 
#                                             logl = {'new': logl_bsbn_fft_skde, 'ref': logl_ref})
#                     except Exception as e:
#                         print(f'Exception in BSBN-FFT-SKDE: {e}')
#                         controller2.pop()
#                         controller.pop()
#                         controller6.pop()
#                         continue

#                     controller5.append(i)

#                     # bsbn-fft-skde linear
#                     try:
#                         vlFT_linear = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=True, use_fft=True)
#                         bsbn_fft_skde_linear, train_time_bsbn_fft_skde_linear, test_time_bsbn_fft_skde_linear, logl_bsbn_fft_skde_linear  = ExperimentsController.train_model('BSBN-FFT-SKDE-Linear', 
#                                                                                                                                             traindat, testdat, pool, vlFT_linear, nodes, patience, parents,
#                                                                                                                                             **{'linear': True, 'FFT': True, 'M': M})
#                         controller7.prepare_dags(bsbn_fft_skde_linear, model_ref)
#                         controller7.append(i, times = {'train_new': train_time_bsbn_fft_skde_linear, 'test_new': test_time_bsbn_fft_skde_linear, 
#                                                     'train_ref': -1, 'test_ref': test_time_ref}, 
#                                             logl = {'new': logl_bsbn_fft_skde_linear, 'ref': logl_ref})
#                     except Exception as e:
#                         print(f'Exception in BSBN-FFT-SKDE-Linear: {e}')
#                         controller2.pop()
#                         controller.pop()
#                         controller6.pop()
#                         controller4.pop()
#                         continue
#                 i+=1
            
#             # Saving results
#             json_DAGS2 = controller2.jsonify(n, f'simu{simu_key}', 'SPBN', 'REF')
            
#             json_DAGS = controller.jsonify(n, f'simu{simu_key}', 'BSBN', 'REF')
#             json_DAGS6 = controller6.jsonify(n, f'simu{simu_key}', 'BSBN-Linear', 'REF')

#             json_DAGS4 = controller4.jsonify(n, f'simu{simu_key}', 'BSBN-FFT-SKDE', 'REF')
#             json_DAGS7 = controller7.jsonify(n, f'simu{simu_key}', 'BSBN-FFT-SKDE-Linear',  'REF')

#             json_DAGS3 = controller3.jsonify(n, f'simu{simu_key}', 'BSBN-FFT', 'REF')
#             json_DAGS5 = controller5.jsonify(n, f'simu{simu_key}', 'BSBN-FFT-Linear', 'REF')


#             # Creating DataFrames and calculating the mean grouped by 'model'
#             res = pd.DataFrame(json_DAGS[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
#             res2 = pd.DataFrame(json_DAGS2[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
#             res3 = pd.DataFrame(json_DAGS3[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
#             res4 = pd.DataFrame(json_DAGS4[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
#             res5 = pd.DataFrame(json_DAGS5[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
#             res6 = pd.DataFrame(json_DAGS6[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
#             res7 = pd.DataFrame(json_DAGS7[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)


#             # Concatenate all results
#             all_res = pd.concat([res2, res, res6, res4, res7, res3, res5])
#             # Print the concatenated results
#             print(all_res)
            
#             svpath = f'results/exp_simu8/Mfix'
#             if not os.path.exists(svpath):
#                 os.makedirs(svpath)


#             all_results = {'1.SPBN': json_DAGS2, 
#                            '2.BSBN': json_DAGS,  '3.BSBN-Linear': json_DAGS6,  
#                            '4.BSBN-FFT-SKDE': json_DAGS4,  '5.BSBN-FFT-SKDE-Linear': json_DAGS7,
#                            '6.BSBN-FFT': json_DAGS3, '7.BSBN-FFT-Linear': json_DAGS5}
            

#             with open(svpath +f'/4-16ksimu_all_{M}.json', 'w') as json_file:
#                 json.dump(all_results, json_file)



            # with open(svpath +f'/{M}/simu_spbn.json', 'w') as json_file:
            #     json.dump(json_DAGS2, json_file)



            # with open(svpath +f'/{M}/simu_bsbn.json', 'w') as json_file:
            #     json.dump(json_DAGS, json_file)

            # with open(svpath +f'/{M}/simu_bsbn_linear.json', 'w') as json_file:
            #     json.dump(json_DAGS6, json_file)
        



            # with open(svpath +f'/{M}/simu_bsbn_fft_skde.json', 'w') as json_file:
            #     json.dump(json_DAGS4, json_file)

            # with open(svpath +f'/{M}/simu_bsbn_fft_skde_linear.json', 'w') as json_file:
            #     json.dump(json_DAGS7, json_file)




            # with open(svpath +f'/{M}/simu_bsbn_fft.json', 'w') as json_file:
            #     json.dump(json_DAGS3, json_file)

            # with open(svpath +f'/{M}/simu_bsbn_fft_linear.json', 'w') as json_file:
            #     json.dump(json_DAGS5, json_file)



