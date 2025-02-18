import numpy as np
import pybnesian as pbn
from .util_syntethic import *
import time


class ExperimentsController:
    def __init__(self, json, nodes, iters, data=None):

        self.data = data
        self.nodes = nodes
        self.simulate = False

        self.nodemap = {}
        for nc in range(len(nodes)):
            self.nodemap[nodes[nc]] = nc
            self.nodemap[nc] = nodes[nc]
        
        self.dag_list_new_model = []
        self.dag_list_ref_model = []
        self.nodetypes_new_model = []
        self.nodetypes_ref_model = []
        self.hamming_np = np.zeros(shape=(iters,))
        self.shamming_np = np.zeros(shape=(iters,))
        self.thamming = np.zeros(shape=(iters,))
        self.error_np = np.zeros(shape=(iters,2))
        self.times_np = np.zeros(shape=(iters,4))
        self.slogls_np = np.zeros(shape=(iters,2))
        self.json_experiment = json
    
        
    def set_up(self, ntrain, ntest, 
               simulate:dict , seeds:tuple = (0,1)):
        
        if not simulate['bool']:
            train = self.data.sample(ntrain, random_state=seeds[0])
            if (self.data.shape[0]-ntrain) < ntest:
                ntest2 = ntest - (self.data.shape[0]-ntrain)
                test1 = self.data.drop(train.index)
                test2 = train.sample(ntest2, random_state=seeds[1])
                test = pd.concat([test1,test2])
            else:
                test = self.data.drop(train.index).sample(ntest, random_state=seeds[1])
        else:
            self.simulate = simulate['bool']
            train = generate_data(simulate['key'],ntrain, seed=seeds[0])
            test = generate_data(simulate['key'],ntest, seed=seeds[1])

            config = get_config(simulate['key'])
            self.ref_model = pbn.SemiparametricBN(nodes = self.nodes, **config)

        return train, test
    
    def get_simulate_ref(self):
        if self.simulate:
            return self.ref_model
    
    def prepare_dags(self, new_model, ref_model):
        
        self.arcs_new_model = new_model.arcs()
        self.nodes_new_model= new_model.node_types()
        self.dag_matrix_new_model = arcs_to_DAG(self.arcs_new_model, self.nodemap)

        self.arcs_ref_model = ref_model.arcs()
        self.nodes_ref_model = ref_model.node_types()
        self.dag_matrix_ref_model = arcs_to_DAG(self.arcs_ref_model, self.nodemap)
        
    
    def append(self, i, times:dict = None, logl:dict = None):
        
        if logl is not None:
            self.dag_list_new_model.append(self.dag_matrix_new_model.tolist())
            self.dag_list_ref_model.append(self.dag_matrix_ref_model.tolist())
            self.nodetypes_new_model.append(ExperimentsController.stringify(self.nodes_new_model.items()))
            self.nodetypes_ref_model.append(ExperimentsController.stringify(self.nodes_ref_model.items()))
                
            self.hamming_np[i] = hamming_distance(self.arcs_new_model, self.arcs_ref_model, self.nodemap)
            self.shamming_np[i] = structural_hamming_distance(self.arcs_new_model, self.arcs_ref_model)
            self.thamming[i] = node_type_hamming_distance(self.nodes_new_model, self.nodes_ref_model)
            self.times_np[i,0] = times['train_new']
            self.times_np[i,1] = times['test_new']
            self.times_np[i,2] = times['train_ref']
            self.times_np[i,3] = times['test_ref']
            self.error_np[i,0] = rmse(logl['ref'], logl['new'])
            self.error_np[i,1] = relative_error(logl['ref'], logl['new'])*100
            self.slogls_np[i,0] = np.sum(logl['new'])
            self.slogls_np[i,1] = np.sum(logl['ref'])
        
        else:
            self.dag_list_new_model.append([])
            self.dag_list_ref_model.append([])
            self.nodetypes_new_model.append([])
            self.nodetypes_ref_model.append([])
            self.hamming_np[i] = -1
            self.shamming_np[i] = -1
            self.thamming[i] = -1
            self.times_np[i] = [-1, -1, -1, -1]
            self.error_np[i] = [-1, -1]
            self.slogls_np[i] = [-1, -1]
    
    def pop(self):
        self.dag_list_new_model.pop()
        self.dag_list_ref_model.pop()
        self.nodetypes_new_model.pop()  
        self.nodetypes_ref_model.pop()
        
            
        
    def jsonify(self, size, span, KEY_NEW, KEY_REF):

        if size not in self.json_experiment:
            self.json_experiment[size] = {f'{span}': {KEY_NEW:{}, KEY_REF:{}, 'dataframe':{}, 'nodemap': self.nodemap}}
        else:
            self.json_experiment[size][f'{span}'] = {KEY_NEW:{}, KEY_REF:{}, 'dataframe':{}, 'nodemap': self.nodemap, }


        self.json_experiment[size][f'{span}'][KEY_NEW] = {'DAGS': self.dag_list_new_model.copy(), 'node_types': self.nodetypes_new_model.copy()}
        self.json_experiment[size][f'{span}'][KEY_REF] = {'DAGS': self.dag_list_ref_model.copy(), 'node_types': self.nodetypes_ref_model.copy()}
        self.json_experiment[size][f'{span}']['dataframe'] = {
        'hamming': self.hamming_np.tolist().copy(), 'shamming': self.shamming_np.tolist().copy(), 'thamming': self.thamming.tolist().copy(),
        'rmse': self.error_np[:,0].tolist().copy(), 'rmae': self.error_np[:,1].tolist().copy(), 
        'train': self.times_np[:,0].tolist().copy(), f'train_{KEY_REF}': self.times_np[:,2].tolist().copy(), 
        'test': self.times_np[:,1].tolist().copy(), f'test_{KEY_REF}': self.times_np[:,3].tolist().copy(),
        'slogl': self.slogls_np[:,0].tolist().copy(), f'slogl_{KEY_REF}': self.slogls_np[:,1].tolist().copy(),
        'model': [KEY_NEW]*len(self.hamming_np)
        }

        return self.json_experiment
    

    @staticmethod
    def stringify(x):
        return [[k,str(v)] for k,v in x] 
    

    @staticmethod
    def train_model(model_key, traindat, testdat, pool, score, nodes, patience, hc_config:dict, **kwargs):
        
        hc = pbn.GreedyHillClimbing()

        if model_key[2:8] == 'B-SPBN':
            start_model = pbn.BinnedSPBN(nodes=nodes)

            start = time.time()
            model = hc.estimate(pool, score, start_model, patience = patience, **hc_config)
            end = time.time()
            train_time = end - start
            model.fit(traindat, **kwargs['args'])    
        
        elif model_key[2:] == "SPBN":
            start_model = pbn.SemiparametricBN(nodes=nodes)

            start = time.time()
            model = hc.estimate(pool, score, start_model, patience = patience, **hc_config)
            end = time.time()
            train_time = end - start
            model.fit(traindat)
            
        elif 'GBN' in model_key:
            start_model = pbn.GaussianNetwork(nodes=nodes)

            start = time.time()
            model = hc.estimate(pool, score, start_model, patience = patience, **hc_config)
            end = time.time()
            train_time = end - start
            model.fit(traindat)

        start = time.time()
        logl_model = model.logl(testdat)
        end = time.time()
        test_time = end - start

        return model, train_time, test_time, logl_model
    

    @staticmethod
    def get_BSPBN_ref(simu_key, traindat, testdat, **kwargs):

        config = ExperimentsController.map_ckde_to_fbkernel(get_config(simu_key))
        model = pbn.BinnedSPBN(**config)
        model.fit(traindat, **kwargs)

        start = time.time()
        logl_model = model.logl(testdat)
        end = time.time()
        test_time = end - start

        return model, test_time, logl_model
        


    @staticmethod
    def map_ckde_to_fbkernel(config):
        # Create a copy of the config to avoid modifying the original
        updated_config = config.copy()
        # Iterate over the node_types and replace CKDEType with FBKernelType
        updated_node_types = [
            (node, pbn.FBKernelType()) if node_type == pbn.CKDEType() else (node, node_type)
            for node, node_type in config['node_types']]
        updated_config['node_types'] = updated_node_types

        return updated_config
    


class ExperimentsReader:
    def __init__(self, results, KEY_NEW, KEY_REF):
        
        self.results = results
        self.nodemap = self.results['nodemap']
        self.KEY_NEW = KEY_NEW
        self.KEY_REF = KEY_REF

    def return_average_dag(self, KEY, threshold, indexes = None):
        map_types = {0: pbn.LinearGaussianCPDType(), 1:pbn.CKDEType(), 2:pbn.FBKernelType()}

  
        network = self.results[KEY]
        if indexes is not None:
            nodetypes = np.array(network['node_types'])[indexes].reshape(-1, 2)
        else:
            nodetypes = np.array(network['node_types']).reshape(-1, 2)
        nodes_df = pd.DataFrame(nodetypes, columns=['node', 'type'])

        nodes_df.replace({'LinearGaussianFactor':0, 'CKDEFactor':1, 'FBKernelType':2}, inplace=True)

        nodes_df = nodes_df.groupby('node').mean().reset_index()
        
        if  KEY[2:8] == 'B-SPBN':
            for idx, row in nodes_df.iterrows():
                if 2 - row['type']  >  row['type']:
                    nodes_df.iloc[idx,1] = 0
                else:
                    nodes_df.iloc[idx,1] = 2

        
        nodes_df.type = nodes_df.type.astype(int)
        avg_node_types = [(row.node, map_types[row.type])  for row in nodes_df.itertuples()]
        avg_nodes = nodes_df['node'].values

        dags = np.array(network['DAGS'])
        if indexes is not None:
            dags = dags[indexes]
        avg_network = average_dags(dags,threshold=threshold)

        lengnodes = avg_network.shape[0]
        avg_arcs = []
        for i in range(lengnodes):
            for j in range(lengnodes):
                val_ij = avg_network[i,j]
                val_ji = avg_network[j,i]
                if i!=j and (val_ij>0 and val_ji==0):
                    avg_arcs.append((self.nodemap[str(i)], self.nodemap[str(j)]))
                elif (val_ij>0 and val_ji>0):
                    print((self.nodemap[str(i)], self.nodemap[str(j)]))
        
        network_kargs = {'node_types':avg_node_types, 'arcs':avg_arcs, 'nodes':avg_nodes}
        return network_kargs
    
    def return_dataframe(self, simulated = False):

        resultsdf = self.results['dataframe']
        resultsdf = pd.DataFrame(resultsdf)

        if simulated:
         
            resultsdf = resultsdf.drop([f'train_{self.KEY_REF}'], axis=1)
            
            
        else:
            resultsdf['ratio_train'] = resultsdf[f'train_{self.KEY_REF}']/resultsdf[f'train_{self.KEY_NEW}']
            resultsdf['ratio_test'] = resultsdf[f'test_{self.KEY_REF}']/resultsdf[f'test_{self.KEY_NEW}']

        return resultsdf



def rmse(logl_est, logl_true):
    """
    Compute the root mean squared error (RMSE) between two log-likelihoods.

    Parameters:
    logl_est (float): Estimated log-likelihood.
    logl_true (float): True log-likelihood.

    Returns:
    float: The RMSE between the two log-likelihoods.
    """
    return np.sqrt(np.mean((logl_est - logl_true) ** 2))

def relative_error(logl_est, logl_true):
    """
    Compute the relative error between two log-likelihoods.

    Parameters:
    logl_est (float): Estimated log-likelihood.
    logl_true (float): True log-likelihood.

    Returns:
    float: The relative error between the two log-likelihoods.
    """
    return np.mean(np.abs(logl_est - logl_true) / np.abs(logl_true))

def sumlogl(logl):
    """
    Compute the sum of log-likelihoods.

    Parameters:
    logl (numpy.ndarray): Array of log-likelihoods.

    Returns:
    float: The sum of log-likelihoods.
    """
    return np.sum(logl)


def hamming_distance(arcs1, arcs2, node_map):
    """
    Compute the Hamming distance between two graphs represented as lists of arcs.

    Parameters:
    arcs1 (list of tuples): List of arcs (edges) in the first graph.
    arcs2 (list of tuples): List of arcs (edges) in the second graph.
    num_nodes (int): Number of nodes in the graphs.

    Returns:
    int: The Hamming distance between the two graphs.
    """
    # Convert arcs to adjacency matrices
    graph1 = arcs_to_adjacency_matrix(arcs1, node_map)
    graph2 = arcs_to_adjacency_matrix(arcs2, node_map)

    # Compute the Hamming distance between the adjacency matrices
    hamming_dist = np.sum(np.abs(graph1 - graph2))

    return hamming_dist / 2

def structural_hamming_distance(arcs1, arcs2):
    """
    Compute the structural Hamming distance between two directed acyclic graphs (DAGs)
    represented as lists of arcs.

    Parameters:
    arcs1 (list of tuples): List of arcs (edges) in the first graph.
    arcs2 (list of tuples): List of arcs (edges) in the second graph.

    Returns:
    int: The structural Hamming distance between the two graphs.
    """
    # Convert lists of arcs to sets for efficient comparison
    arcs_set1 = set(arcs1)
    arcs_set2 = set(arcs2)


    hamming_dist = 0
    for arc1 in arcs1:
            if arc1 not in arcs_set2 and (arc1[1], arc1[0]) in arcs_set2: 
                hamming_dist += 1 # inverse arc 
            elif arc1 not in arcs_set2 and (arc1[1], arc1[0]) not in arcs_set2: 
                hamming_dist += 1 # removal arc 

    for arc2 in arcs2:
        if arc2 not in arcs_set1 and (arc2[1], arc2[0]) not in arcs_set1:
            hamming_dist += 1 # addition arc

    return hamming_dist


def node_type_hamming_distance(node_types, node_types_ref):
    distance = 0

    for k,v in node_types.items():
        if v != node_types_ref[k]:
            if v==pbn.FBKernelType() and node_types_ref[k]==pbn.CKDEType():
                continue
            distance += 1
    
    return distance

def arcs_to_DAG(arcs, node_map):
    """
    Convert a list of arcs to an adjacency matrix.

    Parameters:
    arcs (list of tuples): List of arcs (edges) in the graph, represented as tuples of letters.
    node_map (dict): Mapping from letters to integer node identifiers.

    Returns:
    numpy.ndarray: Adjacency matrix of the graph.
    """
    num_nodes = len(node_map)
    dag_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for arc in arcs:
        # Map letters to integer node identifiers
        i, j = node_map[arc[0]], node_map[arc[1]]
        dag_matrix[i, j] = 1 # arrow from i to j  

    return dag_matrix

def average_dags(dag_list, threshold=0.5):
    """
    Compute a binary average adjacency matrix from a list of DAG adjacency matrices,
    using a threshold to decide the presence of edges.

    :param adjacency_matrices: List of numpy arrays representing adjacency matrices.
    :param threshold: Threshold to decide the presence of an edge in the final matrix.
                      The default value is 0.5.
    :return: Binary adjacency matrix as a numpy array.
    """
    # Check if the list is not empty
    if  dag_list.shape[0] == 0:
        raise ValueError("The list of adjacency matrices is empty.")

    # Sum all adjacency matrices
    summed_matrix = np.sum(dag_list, axis=0)

    # Compute the average by dividing by the number of matrices
    average_matrix = summed_matrix / dag_list.shape[0]

    # Apply the threshold to obtain a binary matrix
    binary_matrix = (average_matrix >= threshold).astype(int)

    return binary_matrix



def arcs_to_adjacency_matrix(arcs, node_map):
    """
    Convert a list of arcs to an adjacency matrix.

    Parameters:
    arcs (list of tuples): List of arcs (edges) in the graph, represented as tuples of letters.
    node_map (dict): Mapping from letters to integer node identifiers.

    Returns:
    numpy.ndarray: Adjacency matrix of the graph.
    """
    num_nodes = len(node_map)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for arc in arcs:
        # Map letters to integer node identifiers
        i, j = node_map[arc[0]], node_map[arc[1]]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # For undirected graphs

    return adj_matrix

