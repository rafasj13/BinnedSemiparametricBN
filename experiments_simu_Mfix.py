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
            

init, fini, iters = (11, 15, 5)
kcv, patience = (5, 3)
simu = True
ntest = 2048
jsondags = {}

# Define all controller configurations and their labels
controller_config = {
    '1.SPBN': {'controller': {}, 'key': 'SPBN', 'args': {'linear': False, 'use_fft': False}},
    '2.B-SPBN-Simple': {'controller': {}, 'key': 'B-SPBN-Simple', 'args': {'linear': False, 'use_fft': False}},
    '3.B-SPBN-Linear': {'controller': {}, 'key': 'B-SPBN-Linear', 'args': {'linear': True, 'use_fft': False}},
    '4.B-SPBN-FKDE-Simple': {'controller': {}, 'key': 'B-SPBN-FKDE-Simple', 'args': {'linear': False, 'use_fft': True}},
    '5.B-SPBN-FKDE-Linear': {'controller': {}, 'key': 'B-SPBN-FKDE-Linear', 'args': {'linear': True, 'use_fft': True}},
}

modelkey = '1.SPBN'

configex = ([[50,80,100, 125]], [[1,2,3,4]], [[11,12,13,14]], ['try1']) # M, simu_key, power, name
for kexp, (grids, simulations, powers, name) in enumerate(zip(*configex)):
    for power in powers:
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
                    
                    traindat, testdat, pool = controller_config[modelkey]['controller'][M].set_up(
                        n, ntest, simulate={'bool': simu, 'key': simu_key}, seeds=(0, 255)
                    )
                    print(i, '->', traindat.shape, testdat.shape)

                    # Reference model
                    model_ref = controller_config[modelkey]['controller'][M].get_simulate_ref()
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
                            if key[2:] =="SPBN":
                                vl = pbn.ValidatedLikelihood(traindat, k=kcv)
                            elif key[0] in ["4", "5"] and simu_key not in [3, 4]:
                                    config['controller'][M].append(i)
                                    continue
                            else:
                                vl = pbn.ValidatedLikelihoodFT(traindat, k=kcv, **configcp['args'])

                            model, train_time, test_time, logl = ExperimentsController.train_model(
                                key, traindat, testdat, pool, vl, nodes, patience, {'max_indegree':parents}, **configcp)

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

                
                svpath = f'results/exp_simu'
                ndirs = len(os.listdir(svpath))+1
                os.makedirs(svpath+f'/{ndirs}Mfix_{name}', exist_ok=True)

                all_results = {key: config['controller'][M] for key,config in controller_config.items()}
                with open(f'{svpath}/simu_all_{M}.json', 'w') as json_file:
                    json.dump(all_results, json_file)
