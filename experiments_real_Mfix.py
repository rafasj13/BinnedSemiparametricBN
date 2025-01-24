import pandas as pd
import numpy as np
import pybnesian as pbn
import os
import time
from utils.util_draw import * 
from utils.util_metrics import *
import json
import glob
import os


 
iters =  5
kcv, patience = (5, 3)
ntest = 2048
jsondags = {}

# Define all controller configurations and their labels
controller_config = {
    '1.SPBN': {'controller': {}, 'key': 'SPBN', 'args': {'linear': False, 'use_fft': False}},
    '2.BSBN': {'controller': {}, 'key': 'BSBN', 'args': {'linear': False, 'use_fft': False}},
    # '3.BSBN-Linear': {'controller': {}, 'key': 'BSBN-Linear', 'args': {'linear': True, 'use_fft': False}},
    '4.BSBN-FKDE': {'controller': {}, 'key': 'BSBN-FKDE', 'args': {'linear': False, 'use_fft': True}},
    '5.GBN-BIC': {'controller': {}, 'key': 'GBN-BIC', 'args': {'linear': False, 'use_fft': False}},
    '6.GBN-BGe': {'controller': {}, 'key': 'GBN-BGe', 'args': {'linear': False, 'use_fft': False}},
    # '5.BSBN-FKDE-Linear': {'controller': {}, 'key': 'BSBN-FKDE-Linear', 'args': {'linear': True, 'use_fft': True}},
    # '6.BSBN-FKDE-SBK': {'controller': {}, 'key': 'BSBN-FKDE-SBK', 'args': {'linear': False, 'use_fft': True}},
    # '7.BSBN-FKDE-SBK-Linear': {'controller': {}, 'key': 'BSBN-FKDE-SBK-Linear', 'args': {'linear': True, 'use_fft': True}},
}

modelkey = '1.SPBN'
parents = None
datasets  = sorted(glob.glob('data/public_datasets/both_ready/*.csv')) 
print(datasets)
configex = ([[50,100]], [datasets], [[14]], ['reals_no_pa']) # M, datasets, power, name
for (grids, paths, powers, name) in zip(*configex):
    for power in powers:
        for M in grids:
            n = 2**power
            results = []

            for npt, path in enumerate(paths):
                basename = os.path.basename(path)
                data = pd.read_csv(path)
                nodes = list(data.columns)
                if npt > 4:
                    continue

                if parents:
                    hc_config = {'max_indegree':parents}
                else:
                    hc_config = {}
                
                print(f'\nSample size ({n},{data.shape[1]}), data {basename}, Grid size {M}, parents {hc_config}')
                # Initialize controllers dynamically
                for key in controller_config:
                    if M not in controller_config[key]['controller']:
                        controller_config[key]['controller'][M] = {}
                    controller = controller_config[key]['controller'][M]
                    controller_config[key]['controller'][M] = ExperimentsController(controller, nodes, iters, data=data)

                i = 0
                while i < iters:
        
                    traindat, testdat = controller_config[modelkey]['controller'][M].set_up(
                        n, ntest, simulate={'bool': False}, seeds=(1, 255)
                    )
                    print(i, '->', traindat.shape, testdat.shape)
                    
                    # Reference model - SPBN
                    configcp = controller_config[modelkey].copy()
                    vl = pbn.ValidatedLikelihood(traindat, k=kcv)
                    pool = pbn.OperatorPool([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()])
                    model_ref, train_time_ref, test_time_ref, logl_ref = ExperimentsController.train_model(
                    modelkey, traindat, testdat, pool, vl, nodes, patience, hc_config, **configcp)

                    

                    # Iterate through configurations
                    for nc,(key, config) in enumerate(controller_config.items()):
                        # try:
                            configcp = config.copy()
                            configcp['args']['grid'] = M
                            if "SPBN" in key:
                                
                                config['controller'][M].prepare_dags(model_ref, model_ref)
                                config['controller'][M].append(
                                i, times={'train_new': train_time_ref, 'test_new': test_time_ref, 'train_ref': train_time_ref, 'test_ref': test_time_ref},
                                   logl={'new': logl_ref, 'ref': logl_ref} )

                                exceptbool = False
                                continue
                            if key[0] in ["4"]:
                                if parents is None or parents > 1:
                                    print('here')
                                    config['controller'][M].append(i)
                                    continue
                                else:
                                    vl = pbn.ValidatedLikelihoodFT(traindat, k=kcv, **configcp['args'])
                                    pool = pbn.OperatorPool([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()])
                            elif key[0] in ["5"]:
                                vl = pbn.BIC(traindat)
                                pool = pbn.OperatorPool([pbn.ArcOperatorSet()])
                            elif key[0] in ["6"]:
                                vl = pbn.BGe(traindat)
                                pool = pbn.OperatorPool([pbn.ArcOperatorSet()])
                            elif key[0] in ["2"]:
                                vl = pbn.ValidatedLikelihoodFT(traindat, k=kcv, **configcp['args'])
                                pool = pbn.OperatorPool([pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()])
                                
                            model, train_time, test_time, logl = ExperimentsController.train_model(
                            key, traindat, testdat, pool, vl, nodes, patience, hc_config, **configcp)

                            config['controller'][M].prepare_dags(model, model_ref)
                            config['controller'][M].append(
                            i, times={'train_new': train_time, 'test_new': test_time, 'train_ref': train_time_ref, 'test_ref': test_time_ref},
                                logl={'new': logl, 'ref': logl_ref})

                            exceptbool = False
                        # except Exception as e:
                        #     print(f'Exception in {key}: {e}')
                        #     for j in range(nc):
                        #         controlist = list(controller_config.keys())
                        #         controller_config[controlist[j]]['controller'][M].pop()

                        #     exceptbool = True
                        #     break

                    i = i+1 if not exceptbool else i

                # Save results dynamically
                for key, config in controller_config.items():   
                    controller_config[key]['controller'][M] = config['controller'][M].jsonify(n, f'real{npt}', key, 'REF')
                    
                # Aggregate and save all results
                all_res = pd.concat([
                    pd.DataFrame(config['controller'][M][n][f'real{npt}']['dataframe'], index=range(iters)).groupby('model').mean().round(4)
                    for config in controller_config.values()])
                print(all_res)

                
                svpath = f'results/exp_real/Mfix'
                ndirs = len(os.listdir(svpath))+1
                os.makedirs(svpath+f'/{ndirs}try_{name}', exist_ok=True)

                all_results = {key: config['controller'][M] for key,config in controller_config.items()}
                with open(f'{svpath}/simu_all_{M}.json', 'w') as json_file:
                    json.dump(all_results, json_file)
