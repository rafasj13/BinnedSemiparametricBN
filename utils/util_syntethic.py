import numpy as np  
import pandas as pd
import pybnesian as pbn



config_synthetic1 = dict(arcs = [('A', 'B'),('A', 'C'),('B', 'D'),('C', 'D'), ('C','E'), ('C', 'G'),
                                ('D', 'E'),('D', 'F'),('E', 'F'),('A', 'F')],

    node_types = [('A', pbn.LinearGaussianCPDType()), ('B', pbn.LinearGaussianCPDType()), ('C', pbn.CKDEType()), ('D', pbn.CKDEType()),
                  ('E', pbn.CKDEType()), ('F', pbn.CKDEType()), ('G', pbn.LinearGaussianCPDType())])


config_synthetic2 = dict(
    arcs=[
        # Linear Gaussian to CKDE (prioritize CKDE as children)
        ('A', 'B'), ('A', 'C'), ('A', 'D'),  # A has 3 children
        ('C', 'E'), ('C', 'F'),  # C has 2 children
        ('D', 'G'), ('D', 'H'),  # D has 2 children

        # CKDE to Linear Gaussian
        ('B', 'I'), ('E', 'J'), ('F', 'K'),  # B, E, F each have 1 child

        # Allowing 4-5 parents for some nodes
        ('A', 'L'), ('C', 'L'), ('D', 'L'), ('F', 'L'), ('H', 'L'),  # L has 5 parents
        ('B', 'M'), ('E', 'M'), ('G', 'M'), ('J', 'M'),  # M has 4 parents

        # Additional arcs to ensure full connectivity
        ('H', 'F'), ('J', 'G')
    ],
    node_types=[
        # CKDE nodes
        ('B', pbn.CKDEType()), ('C', pbn.CKDEType()), ('E', pbn.CKDEType()), 
        ('F', pbn.CKDEType()), ('G', pbn.CKDEType()), ('H', pbn.CKDEType()), 
        ('L', pbn.CKDEType()), ('M', pbn.CKDEType()),

        # Linear Gaussian nodes
        ('A', pbn.LinearGaussianCPDType()), ('D', pbn.LinearGaussianCPDType()), 
        ('I', pbn.LinearGaussianCPDType()), ('J', pbn.LinearGaussianCPDType()), 
        ('K', pbn.LinearGaussianCPDType())
    ]
)


config_synthetic3 = dict(arcs = [('A', 'B'),('B', 'C'),('B', 'D'),('D', 'E'),('D', 'F'),('C', 'G'),('C', 'H')],
    node_types = [('A', pbn.CKDEType()), ('B', pbn.LinearGaussianCPDType()), ('C', pbn.LinearGaussianCPDType()), ('D', pbn.CKDEType()),
                  ('E', pbn.CKDEType()), ('F', pbn.CKDEType()), ('G', pbn.LinearGaussianCPDType()), ('H', pbn.CKDEType())])

config_synthetic4 = dict(
    arcs = [('A', 'B'),('A', 'C'),
            ('B', 'D'),
            ('C', 'E'),('C', 'F'),
            ('D', 'G'),('D', 'H'),('D', 'K'),
            ('E', 'I'),('E', 'J'),
            ('F', 'O'),
            ('J', 'M'),('J', 'N'),
            #('H', 'K'), 
            ('H', 'L')],

 node_types= [('A', pbn.LinearGaussianCPDType()), ('B', pbn.LinearGaussianCPDType()), ('C', pbn.CKDEType()),
               ('D', pbn.CKDEType()),('E', pbn.LinearGaussianCPDType()), ('F', pbn.CKDEType()), 
               ('G', pbn.CKDEType()), ('H', pbn.LinearGaussianCPDType()), ('I', pbn.CKDEType()), ('J', pbn.LinearGaussianCPDType()), 
               ('K', pbn.LinearGaussianCPDType()), ('L', pbn.CKDEType()),
            ('M', pbn.CKDEType()), ('N', pbn.CKDEType()), ('O', pbn.CKDEType())])


def sample_mixture(prior_prob, means, variances, n_instances,):

    p = np.asarray(prior_prob)
    c = np.cumsum(p)
    m = np.asarray(means)
    v = np.asarray(variances)

    s = np.random.uniform(size=n_instances)

    digitize = np.digitize(s, c) 

    res = np.random.normal(m[digitize], np.sqrt(v[digitize]))

    return res



def generate_synthetic1(size, seed = 0):
  np.random.seed(seed)        
  
  datarray = np.zeros(shape=(size,7))
  for row in range(size):
    
    a = np.random.normal(3, 2)

    b = np.random.normal(a*0.5, 2)
    # b = sample_mixture([0.5, 0.5], [a-5, 0], np.array([1.5, 1])**2, 1)[0]
    c = sample_mixture([0.45, 0.55], [a*0.5, 5], np.array([1.5, 1])**2, 1)[0]
    # d = np.random.normal(b*0.3+c, 1)
    d = sample_mixture([0.5, 0.5], [c*b*0.5, 3.5], np.array([1, 1])**2, 1)[0]
    # sample_mixture([0.3, 0.7], [a-c, a*0.3+c], np.array([1, 0.5])**2, 1)[0] 
    e = sample_mixture([0.5, 0.5], [d+c, 2], np.array([1, 1])**2, 1)[0]  
    f = sample_mixture([0.5, 0.5], [e+d, 0.7*a], np.array([1, 0.5])**2, 1)[0]
    g = np.random.normal(c*0.3, 2)

    datarray[row] = [a,b,c,d,e,f,g]

  return pd.DataFrame(datarray, columns=['A','B','C','D','E','F','G'])

def generate_synthetic2(size, seed=0):
    np.random.seed(seed)
    datarray = np.zeros(shape=(size, 13))
    
    for row in range(size):
        # Linear Gaussian nodes
        a = np.random.normal(4, 1.5)
        b = sample_mixture([0.4, 0.6], [a * 1.2, 1], np.array([1.1, 1])**2, 1)[0]
        c = sample_mixture([0.5, 0.5], [a + 1, 1], np.array([1.2, 1])**2, 1)[0]
        d = np.random.normal(a * 0.8, 1.3)

        e = sample_mixture([0.6, 0.4], [c * 1.2, -1], np.array([1.3, 1.5])**2, 1)[0]
        h = sample_mixture([0.6, 0.4], [d*2, 0], np.array([1.2, 1.8])**2, 1)[0]
        i = np.random.normal(b * 0.6, 2)

        j = np.random.normal(e * 0.7, 1.7)
        f = sample_mixture([0.5, 0.5], [c * 1.1+h, 15], np.array([1, 1.2])**2, 1)[0]
        
        g = sample_mixture([0.5, 0.5], [d * 0.8+j, 0], np.array([1, 1])**2, 1)[0]
        k = np.random.normal(f * 0.3, 2)
        l = sample_mixture([0.5, 0.5], [a + c + f, h * 0.6+d], np.array([1, 1.5])**2, 1)[0]

        m = sample_mixture([0.4, 0.6], [b + e + g, j * 0.7], np.array([1.2, 1.3])**2, 1)[0]
        
        # Assign values to the array
        datarray[row] = [a, b, c, d, e, f, g, h, i, j, k, l, m]

    # Convert to DataFrame with appropriate column names
    return pd.DataFrame(datarray, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'])

def generate_synthetic3(size, seed = 0):
  np.random.seed(seed)        
  
  datarray = np.zeros(shape=(size,8))
  for row in range(size):
    
    a = sample_mixture([0.5, 0.5], [4, 1], np.array([2, 1])**2, 1)[0]
    b = np.random.normal(a*0.5, 2)

    c = np.random.normal(b*2, 1.5)
    d = sample_mixture([0.5, 0.5], [b-1,10], np.array([1, 1.5])**2, 1)[0]
    

    e = sample_mixture([0.5, 0.5], [d*2, 3], np.array([1.5, 1])**2, 1)[0]  
    f = sample_mixture([0.6, 0.4], [d*1.5, 0], np.array([1.5, 1])**2, 1)[0]
    g = np.random.normal(c*0.3+5, 1)
    h = sample_mixture([0.5, 0.5], [c*0.5, 10], np.array([1, 1])**2, 1)[0]

    datarray[row] = [a,b,c,d,e,f,g,h]

  return pd.DataFrame(datarray, columns=['A','B','C','D','E','F','G','H'])
            
def generate_synthetic4(size, seed=0):
    np.random.seed(seed)
    datarray = np.zeros(shape=(size, 15))
    
    for row in range(size):
        # Generate data for each variable based on its dependencies
        a = np.random.normal(5, 2)
        b = np.random.normal(a + 2, 1.5)
        c = sample_mixture([0.4, 0.6], [a + 2, 1], np.array([1, 1.5])**2, 1)[0]
        
        d = sample_mixture([0.5, 0.5], [b * 0.8, 15], np.array([1.5, 1.5])**2, 1)[0]
        e = np.random.normal(c * 0.7, 2.0)
        f = sample_mixture([0.5, 0.5], [c * 1.2, -3], np.array([1.5, 1])**2, 1)[0]
        
        g = sample_mixture([0.6, 0.4], [d + 4, 8], np.array([1, 1.5])**2, 1)[0]
        h = np.random.normal(d * 0.4, 2)
        k = np.random.normal(d * 0.5, 2.5)
        
        i = sample_mixture([0.55, 0.45], [e * 1.3, 0], np.array([2, 1])**2, 1)[0]
        j = np.random.normal(e * 0.5, 2)
        
        o = sample_mixture([0.3, 0.7], [f + 1, -2], np.array([1.4, 0.7])**2, 1)[0]
        
        m = sample_mixture([0.6, 0.4], [j * 1.5, 7], np.array([1, 1.5])**2, 1)[0]
        n = sample_mixture([0.4, 0.6], [j * 1.1, -1], np.array([1.2, 1.3])**2, 1)[0]
        
        l = sample_mixture([0.5, 0.5], [h * 0.3, 5], np.array([1.1, 1.4])**2, 1)[0]
        
        # Assign values to the array
        datarray[row] = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o]

    # Convert to DataFrame with appropriate column names
    return pd.DataFrame(datarray, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'])



def generate_data(key, size, seed):
    if key==1:
        return generate_synthetic1(size, seed)
    elif key==2:
        return generate_synthetic2(size, seed)
    elif key==3:
        return generate_synthetic3(size, seed)
    elif key==4:
        return generate_synthetic4(size, seed)
    else:
        print('Not valid key')

def get_config(key):
    if key==1:
        return config_synthetic1
    elif key==2:
        return config_synthetic2
    elif key==3:
        return config_synthetic3
    elif key==4:
        return config_synthetic4
    else:
        print('Not valid key')