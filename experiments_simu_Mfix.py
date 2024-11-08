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
    if key == 2:
        return ['A','B','C','D','E','F','G'], 3
    elif key == 3:
        return ['A','B','C','D','E','F','G','H'], 1
    elif key == 4:
        return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'], 1
            
init,fini, iters = (11,15,15)
# M = 75
kcv, patience, max_indegree = (5, 5, 3)
json_DAGS, json_DAGS2, json_DAGS3, json_DAGS4, json_DAGS5, json_DAGS6, json_DAGS7 = {}, {}, {}, {}, {}, {}, {}
simu = True
ntest = 2048


for M in  [30, 50, 80]:
    for power in range(init,fini):
        n=2**power
        results = []

        for simu_key in [2,3,4]:
            nodes, parents = get_nodes_and_parents(simu_key)
            print(f'\nSample size {n}, Simulated data {simu_key}, Grid size {M}')
            print(get_config(simu_key)['arcs'])

            controller2 = ExperimentsController(json_DAGS2, nodes, iters) # spbn

            controller = ExperimentsController(json_DAGS, nodes, iters) # bsbn simple
            controller6 = ExperimentsController(json_DAGS6, nodes, iters) # bsbn linear
            
            controller4 = ExperimentsController(json_DAGS4, nodes, iters) # bsbn-fft-skde simple
            controller7 = ExperimentsController(json_DAGS7, nodes, iters) # bsbn-fft-skde linear

            controller3 = ExperimentsController(json_DAGS3, nodes, iters) # bsbn-fft simple
            controller5 = ExperimentsController(json_DAGS5, nodes, iters) # bsbn-fft linear 


            for i in range(iters):

                traindat, testdat, pool = controller.set_up(n, ntest, simulate = {'bool':simu, 'key':simu_key}, seeds=(0,255))
                print(i,'->',traindat.shape, testdat.shape)
                
                model_ref = controller.get_simulate_ref()
                model_ref.fit(traindat)
                logl_ref = model_ref.logl(testdat)

                # spbn
                vl = pbn.ValidatedLikelihood(traindat, k=kcv)
                spbn, train_time_spbn, test_time_spbn, logl_spbn  = ExperimentsController.train_model('SPBN',
                                                                                        traindat, testdat, pool, vl, nodes, patience, parents)
                controller2.prepare_dags(spbn, model_ref)
                controller2.append(i, times = {'train_new': train_time_spbn, 'test_new': test_time_spbn, 
                                            'train_ref': -1, 'test_ref': -1}, 
                                    logl = {'new': logl_spbn, 'ref': logl_ref})
                
                # bsbn
                vlBin = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=False, use_fft=False)
                bsbn, train_time_bsbn, test_time_bsbn, logl_bsbn  = ExperimentsController.train_model('BSBN', 
                                                                                        traindat, testdat, pool, vlBin, nodes, patience, parents,
                                                                                        **{'linear': False, 'FFT': False, 'M': M})
                controller.prepare_dags(bsbn, model_ref)
                controller.append(i, times = {'train_new': train_time_bsbn, 'test_new': test_time_bsbn, 
                                            'train_ref': -1, 'test_ref': -1}, 
                                    logl = {'new': logl_bsbn, 'ref': logl_ref})
                
                
                # bsbn linear
                vlBin = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=True, use_fft=False)
                bsbn_linear, train_time_bsbn_linear, test_time_bsbn_linear, logl_bsbn_linear  = ExperimentsController.train_model('BSBN-Linear', 
                                                                                                traindat, testdat, pool, vlBin, nodes, patience, parents,
                                                                                                **{'linear': True, 'FFT': False, 'M': M})
                controller6.prepare_dags(bsbn_linear, model_ref)
                controller6.append(i, times = {'train_new': train_time_bsbn_linear, 'test_new': test_time_bsbn_linear, 
                                            'train_ref': -1, 'test_ref': -1}, 
                                    logl = {'new': logl_bsbn_linear, 'ref': logl_ref})

                
                if simu_key in [3,4]:
                    # bsbn-fft
                    vlFT = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=False, use_fft=True)
                    bsbn_fft, train_time_bsbn_fft, test_time_bsbn_fft, logl_bsbn_fft  = ExperimentsController.train_model('BSBN-FFT', 
                                                                                            traindat, testdat, pool, vlFT, nodes, patience, parents,
                                                                                            **{'linear': False, 'FFT': True, 'M': M})
                    controller3.prepare_dags(bsbn_fft, model_ref)
                    controller3.append(i, times = {'train_new': train_time_bsbn_fft, 'test_new': test_time_bsbn_fft, 
                                                'train_ref': -1, 'test_ref': -1}, 
                                        logl = {'new': logl_bsbn_fft, 'ref': logl_ref})
                    
                    controller4.append(i)


                    # bsbn-fft linear
                    vlFT_linear = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=True, use_fft=True)
                    bsbn_fft_linear, train_time_bsbn_fft_linear, test_time_bsbn_fft_linear, logl_bsbn_fft_linear  = ExperimentsController.train_model('BSBN-FFT-Linear', 
                                                                                                                    traindat, testdat, pool, vlFT_linear, nodes, patience, parents,
                                                                                                                    **{'linear': True, 'FFT': True, 'M': M})
                    controller5.prepare_dags(bsbn_fft_linear, model_ref)
                    controller5.append(i, times = {'train_new': train_time_bsbn_fft_linear, 'test_new': test_time_bsbn_fft_linear, 
                                                'train_ref': -1, 'test_ref': -1}, 
                                        logl = {'new': logl_bsbn_fft_linear, 'ref': logl_ref})
                    
                    controller7.append(i)


                else:
                    controller3.append(i)

                    # bsbn-fft-skde
                    vlFT = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=False, use_fft=True)
                    bsbn_fft_skde, train_time_bsbn_fft_skde, test_time_bsbn_fft_skde, logl_bsbn_fft_skde  = ExperimentsController.train_model('BSBN-FFT-SKDE', 
                                                                                                            traindat, testdat, pool, vlFT, nodes, patience, parents,
                                                                                                            **{'linear': False, 'FFT': True, 'M': M})
                    controller4.prepare_dags(bsbn_fft_skde, model_ref)
                    controller4.append(i, times = {'train_new': train_time_bsbn_fft_skde, 'test_new': test_time_bsbn_fft_skde, 
                                                'train_ref': -1, 'test_ref': -1}, 
                                        logl = {'new': logl_bsbn_fft_skde, 'ref': logl_ref})
                    
                    controller5.append(i)

                    # bsbn-fft-skde linear
                    vlFT_linear = pbn.ValidatedLikelihoodFT(traindat, k=kcv, grid_size=M, linear_binning=True, use_fft=True)
                    bsbn_fft_skde_linear, train_time_bsbn_fft_skde_linear, test_time_bsbn_fft_skde_linear, logl_bsbn_fft_skde_linear  = ExperimentsController.train_model('BSBN-FFT-SKDE-Linear', 
                                                                                                                                        traindat, testdat, pool, vlFT_linear, nodes, patience, parents,
                                                                                                                                        **{'linear': True, 'FFT': True, 'M': M})
                    controller7.prepare_dags(bsbn_fft_skde_linear, model_ref)
                    controller7.append(i, times = {'train_new': train_time_bsbn_fft_skde_linear, 'test_new': test_time_bsbn_fft_skde_linear, 
                                                'train_ref': -1, 'test_ref': -1}, 
                                        logl = {'new': logl_bsbn_fft_skde_linear, 'ref': logl_ref})

            
            
            
            json_DAGS2 = controller2.jsonify(n, f'simu{simu_key}', 'SPBN', 'REF')
            
            json_DAGS = controller.jsonify(n, f'simu{simu_key}', 'BSBN', 'REF')
            json_DAGS6 = controller6.jsonify(n, f'simu{simu_key}', 'BSBN-Linear', 'REF')

            json_DAGS4 = controller4.jsonify(n, f'simu{simu_key}', 'BSBN-FFT-SKDE', 'REF')
            json_DAGS7 = controller7.jsonify(n, f'simu{simu_key}', 'BSBN-FFT-SKDE-Linear',  'REF')

            json_DAGS3 = controller3.jsonify(n, f'simu{simu_key}', 'BSBN-FFT', 'REF')
            json_DAGS5 = controller5.jsonify(n, f'simu{simu_key}', 'BSBN-FFT-Linear', 'REF')


            # Creating DataFrames and calculating the mean grouped by 'model'
            res = pd.DataFrame(json_DAGS[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
            res2 = pd.DataFrame(json_DAGS2[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
            res3 = pd.DataFrame(json_DAGS3[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
            res4 = pd.DataFrame(json_DAGS4[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
            res5 = pd.DataFrame(json_DAGS5[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
            res6 = pd.DataFrame(json_DAGS6[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
            res7 = pd.DataFrame(json_DAGS7[n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)


            # Concatenate all results
            all_res = pd.concat([res2, res, res6, res4, res7, res3, res5])
            # Print the concatenated results
            print(all_res)
            
            svpath = f'results/exp_simu7/Mfix/{M}'
            if not os.path.exists(svpath):
                os.makedirs(svpath)

            with open(svpath +'/simu_spbn.json', 'w') as json_file:
                json.dump(json_DAGS2, json_file)




            with open(svpath +'/simu_bsbn.json', 'w') as json_file:
                json.dump(json_DAGS, json_file)

            with open(svpath +'/simu_bsbn_linear.json', 'w') as json_file:
                json.dump(json_DAGS6, json_file)
        



            with open(svpath +'/simu_bsbn_fft_skde.json', 'w') as json_file:
                json.dump(json_DAGS4, json_file)

            with open(svpath +'/simu_bsbn_fft_skde_linear.json', 'w') as json_file:
                json.dump(json_DAGS7, json_file)




            with open(svpath +'/simu_bsbn_fft.json', 'w') as json_file:
                json.dump(json_DAGS3, json_file)

            with open(svpath +'/simu_bsbn_fft_linear.json', 'w') as json_file:
                json.dump(json_DAGS5, json_file)



