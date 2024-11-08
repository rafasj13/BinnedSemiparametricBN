import pandas as pd
import numpy as np
import pybnesian as pbn
import os
import time
from utils.util_draw import * 
from utils.util_metrics import *
import glob
import json


init,fini,jump, iters = (25,225,25,10)
linear, FFT = (False, False)
n, ntest = (2**14 , 2048)
kcv, patience, max_indegree = (5, 5, 5)
simu = False
json_DAGS = {}

csvs = sorted(glob.glob('data/aingura/dfg/*.csv'))
for msize in [125,200]:#range(init,fini,jump):  
    results = []
    for file in np.array(csvs)[[0]]:
            
        namefile = os.path.basename(file).split('.')[0]
        span = namefile.split('_')[1]
        print(f'\n\n{namefile}')

        data = pd.read_csv(file, sep=';')
        print(data)
        print(f'\nGrid size {msize}')

        nodes = list(data.columns[1:]) 
        controller = ExperimentsController(json_DAGS, nodes, iters, data=data)

        for i in range(iters):
    
 
            traindat, testdat, pool, vl, vlFT = controller.set_up(n, ntest, k=kcv, M=msize, linear=linear, FFT=FFT,
                                                               simulate = {'bool':simu, 'key':1}, seeds=(10,15))



            start_bsbn = pbn.FourierNetwork(nodes=nodes)
            hc = pbn.GreedyHillClimbing()
            ## bsbn
            start = time.time()
            bsbn = hc.estimate(pool, vlFT, start_bsbn, patience = patience, max_indegree=max_indegree)
            end = time.time()
            train_time_bsbn = end - start

            bsbn.fit(traindat, grid= msize, linear = linear, use_fft=FFT)    

            start = time.time()
            logl_bsbn = bsbn.logl(testdat)
            end = time.time()
            test_time_bsbn = end - start



            start_spbn = pbn.SemiparametricBN(nodes=nodes)
            hc = pbn.GreedyHillClimbing()
            ## spbn
            start = time.time()
            spbn = hc.estimate(pool, vl, start_spbn, patience = patience, max_indegree=max_indegree)
            end = time.time()
            train_time_spbn = end - start

            spbn.fit(traindat)

            start = time.time()
            logl_spbn = spbn.logl(testdat)
            end = time.time()
            test_time_spbn = end - start


            controller.prepare_dags(bsbn, spbn)
            controller.append(i, times = {'train_new': train_time_bsbn, 'test_new': test_time_bsbn, 
                                          'train_ref': train_time_spbn, 'test_ref': test_time_spbn}, 
                                 logl = {'new': logl_bsbn, 'ref': logl_spbn})
          
        
        json_DAGS = controller.jsonify(msize, span, 'BSBN', 'SPBN')
        res = pd.DataFrame(json_DAGS[msize][f'{span}']['dataframe'], index=range(iters))

        
        print(res,'\n')
        print(res.groupby('grid').mean())
        results.append(res)

        # results_df = pd.concat(results, axis=0).reset_index(drop=True)
        # results_df.to_csv(f'results/{namefile}_Mfixed.csv',index=False)
        
        # svpath = f'results/Nfix/exp7_simple/N{n}'
        # if not os.path.exists(svpath):
        #     os.makedirs(svpath)
        # with open(svpath +'/Nfixed_125200.json', 'w') as json_file:
        #     json.dump(json_DAGS, json_file)
    
    results_df = pd.concat(results, axis=0).reset_index(drop=True)
    print('\n',results)

