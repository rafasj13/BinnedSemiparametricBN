import pandas as pd
import numpy as np
import pybnesian as pbn
import os
import time
from utils.util_draw import * 
from utils.util_metrics import *
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
            

init, fini, iters = (11, 15, 15)
simu = True
ntest = 2048
jsondags = {}

# Define all controller configurations and their labels
controller_config = {
    '1.SPBN': {'controller': {}, 'key': 'SPBN', 'args': {'linear': False, 'use_fft': False}},
    '2.BSBN': {'controller': {}, 'key': 'BSBN', 'args': {'linear': False, 'use_fft': False}},
    '3.BSBN-Linear': {'controller': {}, 'key': 'BSBN-Linear', 'args': {'linear': True, 'use_fft': False}},
    '4.BSBN-FKDE': {'controller': {}, 'key': 'BSBN-FFT', 'args': {'linear': False, 'use_fft': True}},
    '5.BSBN-FKDE-Linear': {'controller': {}, 'key': 'BSBN-FFT-Linear', 'args': {'linear': True, 'use_fft': True}},
    '6.BSBN-FKDE-SBK': {'controller': {}, 'key': 'BSBN-FFT-SKDE', 'args': {'linear': False, 'use_fft': True}},
    '7.BSBN-FKDE-SBK-Linear': {'controller': {}, 'key': 'BSBN-FFT-SKDE-Linear', 'args': {'linear': True, 'use_fft': True}},
}
configex = ([[50,80,100]], [[3,4]], [11]) # M, simu_key, power

for kexp, (grids, simulations, new_init) in enumerate(zip(*configex)):
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
                        configcp = config.copy()
                        configcp['args']['grid'] = M
                        try:
                            if "SPBN" in key:
                                config['controller'][M].prepare_dags(model_ref, model_ref)
                                config['controller'][M].append(
                                i, times={'train_new': -1, 'test_new': test_time_ref, 'train_ref': -1, 'test_ref': test_time_ref},
                                   logl={'new': logl_ref, 'ref': logl_ref}
                                )
                                continue
                            elif key[0] in ["4", "5"] and simu_key not in [3, 4]:
                                config['controller'][M].append(i)
                                continue
                            elif key[0] in ["6", "7"]  and simu_key in [3, 4]:
                                config['controller'][M].append(i)
                                continue
                            
                            model, test_time, logl  = ExperimentsController.get_bsbn_ref(simu_key, traindat, testdat, **configcp['args'])
                            config['controller'][M].prepare_dags(model, model_ref)
                            config['controller'][M].append(
                                i, times={'train_new': -1, 'test_new': test_time, 'train_ref': -1, 'test_ref': test_time_ref},
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

                svpath = f'results/exp_simu/Mfix_sameDAG{kexp+3}'
                os.makedirs(svpath, exist_ok=True)

                all_results = {key: config['controller'][M] for key,config in controller_config.items()}
                with open(f'{svpath}/simu_all_{M}.json', 'w') as json_file:
                    json.dump(all_results, json_file)

