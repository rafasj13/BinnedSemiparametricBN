import pandas as pd
import numpy as np
import pybnesian as pbn
import os
import time
from utils.util_draw import * 
from utils.util_metrics import *
import json

def get_nodes_and_parents(key): #return nodes and parents
    if key == 1 or key==5 or key==7:
        return ['A','B','C','D','E','F','G'], 3
    if key == 2:
        return ['A','B','C','D','E','F','G','H','I','J','K','L','M'], 5
        
    elif key == 3 or key==6 or key==8:
        return ['A','B','C','D','E','F','G','H'], 1
    elif key == 4:
        return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'], 1
            

init, fini, iters = (11, 15, 1)
simu = True
ntest = 2048
jsondags = {}

# Define all controller configurations and their labels
controller_template = {
    '1.SPBN': {'key': 'SPBN', 'args': {'linear': False, 'use_fft': False}},
    '2.B-SPBN-Simple': {'key': 'B-SPBN-Simple', 'args': {'linear': False, 'use_fft': False}},
    '3.B-SPBN-Linear': {'key': 'B-SPBN-Linear', 'args': {'linear': True, 'use_fft': False}},
    '4.B-SPBN-FKDE-Simple': {'key': 'B-SPBN-FFT-Simple', 'args': {'linear': False, 'use_fft': True}},
    '5.B-SPBN-FKDE-Linear': {'key': 'B-SPBN-FFT-Linear', 'args': {'linear': True, 'use_fft': True}},
}

configex = ([[100]], [[1,4]], [[13]], ['sameDAG_noise'])
for kexp, (grids, simulations, powers, name) in enumerate(zip(*configex)):
    for power in powers:
        for M in grids:
            n = 2**power
            for noise_level in [0, 0.1, 0.2, 0.3]:
                # Store all simulations for this noise level in a single JSON.
                all_results_noise = {key: {} for key in controller_template.keys()}

                for simu_key in simulations:
                    nodes, parents = get_nodes_and_parents(simu_key)
                    print(f'\nSample size {n}, Simulated data {simu_key}, Grid size {M}')
                    print(get_config(simu_key)['arcs'])

                    # Fresh controllers for this (noise_level, simu_key) run.
                    controllers = {
                        key: ExperimentsController({}, nodes, iters)
                        for key in controller_template.keys()
                    }

                    i = 0
                    while i < iters:
                        
                        traindat, testdat = controllers['1.SPBN'].set_up(
                            n, ntest, simulate={'bool': simu, 'key': simu_key}, seeds=(1, 255)
                        )
                        
                    
                        traindat = apply_noise(traindat, noise_level=noise_level)
                        testdat = apply_noise(testdat, noise_level=noise_level)
                        print(i, '->', traindat.shape, testdat.shape)

                        # Reference model
                        model_ref = controllers['1.SPBN'].get_simulate_ref()
                        model_ref.fit(traindat)
                        start = time.time()
                        logl_ref = model_ref.logl(testdat)
                        end = time.time()
                        test_time_ref = end-start

                        # Iterate through configurations
                        for nc, (key, config) in enumerate(controller_template.items()):
                            configcp = config.copy()
                            configcp['args'] = config['args'].copy()
                            configcp['args']['grid'] = M
                            
                            try:
                                if key[2:] =="SPBN":
                                    controllers[key].prepare_dags(model_ref, model_ref)
                                    controllers[key].append(
                                    i, times={'train_new': -1, 'test_new': test_time_ref, 'train_ref': -1, 'test_ref': test_time_ref},
                                    logl={'new': logl_ref, 'ref': logl_ref}
                                    )
                                    continue
                                elif (key[0] in ["4", "5"] and simu_key in [2]) or (key[0] in ["4", "5"] and simu_key in [1,5,7] and M>25):
                                    controllers[key].append(i)
                                    continue
                                                
                                model, test_time, logl  = ExperimentsController.get_BSPBN_ref(simu_key, traindat, testdat, **configcp['args'])
                                controllers[key].prepare_dags(model, model_ref)
                                controllers[key].append(
                                    i, times={'train_new': -1, 'test_new': test_time, 'train_ref': -1, 'test_ref': test_time_ref},
                                    logl={'new': logl, 'ref': logl_ref}
                                )
                                
                    
                                exceptbool = False
                            except Exception as e:
                                print(f'Exception in {key}: {e}')
                                for j in range(nc):
                                    controlist = list(controller_template.keys())
                                    controllers[controlist[j]].pop()

                                exceptbool = True
                                break

                        i = i+1 if not exceptbool else i

                    # Save this simulation into the in-memory bucket for this noise.
                    for key in controller_template.keys():
                        model_json = controllers[key].jsonify(n, f'simu{simu_key}', key, 'REF')
                        n_key = next(iter(model_json.keys()))
                        if n_key not in all_results_noise[key]:
                            all_results_noise[key][n_key] = {}
                        all_results_noise[key][n_key].update(model_json[n_key])
                        
                    # Aggregate and save all results
                    all_res = pd.concat([
                        pd.DataFrame(all_results_noise[key][n][f'simu{simu_key}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
                        for key in controller_template.keys()])
                    print(all_res)

                    
                    svpath = f'results/exp_simu/Mfix_{name}'
                    os.makedirs(svpath, exist_ok=True)

                # Write once per noise level (contains both simu1 and simu4).
                with open(f'{svpath}/simu_all_{M}_{noise_level}.json', 'w') as json_file:
                    json.dump(all_results_noise, json_file)

