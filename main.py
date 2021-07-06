

"""
TODO:
    - Train embedding model. 
    - Apply embeddings to data. 
    - Encode data. 
    - Train,valid,test model
"""

from autoencoder import create_auto_encoder
from model import create_model, CorrelelatedFeatures, ApproxKerasSVM, coeff_determination
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from random import shuffle

from collections import defaultdict
import tensorflow as tf

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor, BayesianRidge

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.tree import DecisionTreeRegressor

from itertools import product
from random import choice, choices
from sklearn.pipeline import Pipeline

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA,FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.dummy import DummyRegressor

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score, LeaveOneOut

MAX_ENCODER_EPOCHS = 1000
MAX_EPOCHS = 1000
EPSILON = 1e-10
MODEL = 'ComplEx'
hidden_dim = (128,)
SEED = 42
np.random.seed(SEED)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings('ignore')


def load_fingerprints(filename):
    df = pd.read_csv(filename,index_col='chemical')
    l = len(df.iloc[0]['fingerprint'])
    out = {}
    for c in df.index:
        fp = df.loc[c]['fingerprint']
        v = [int(f) for f in fp]
        out[c] = np.asarray(v)
    return out

def load_features(filename):
    df = pd.read_csv(filename,index_col='chemical')
    df = df.dropna()
    columns = df.columns
    out = {}
    for c in df.index:
        v = [df.loc[c][col] for col in columns]
        out[c] = np.asarray(v)
    return out

def load_one_hot(entities):
    all_entities = list(set(entities))
    out = {}
    for e in entities:
        v = np.zeros((len(all_entities),))
        v[all_entities.index(e)] = 1
        out[e] = np.asarray(v)
    return out
    

def load_embeddings(filename,filename_ids):
    df = np.load(filename)
    ids = dict(np.load(filename_ids))
    
    return {k:df[int(ids[k])] for k in ids}
    
def load_data(filename,filter_chemicals=None, filter_species=None):
    df = pd.read_csv(filename)
    X,y = [],[]
    if filter_chemicals:
        to_drop = set(df.chemical) - filter_chemicals
        for c in to_drop:
            df = df.drop(df[df.chemical == c].index)
    if filter_species:
        to_drop = set(df.species) - filter_species
        for s in to_drop:
            df = df.drop(df[df.species == s].index)
        
    df = df.drop(df[df.study_duration > 24*14].index)
    
    df = df.groupby(['chemical','species'],as_index=False).mean()
    X = list(zip(df['chemical'],df['species']))
    y = np.log(df.concentration+EPSILON)
    
    tmp = np.asarray(df.study_duration).reshape((-1,1))
    mms = StandardScaler()
    tmp = mms.fit_transform(tmp)
    experimental_features = dict(zip(X,tmp.reshape(-1,1)))
    
    y = np.asarray(y).reshape((-1,1))
    #y = MinMaxScaler().fit_transform(y)
    
    return X, y, experimental_features

def data_split(X,Y,restrictions=None,method = 1, variant = 1, prop=0.33):
    """
    C_x - chemical set
    S_x - species set 
    t,v - training,validation
    1. C_t \cap C_v == Ø and S_t \cap S_v != Ø,
    2. C_t \cap C_v == Ø and S_t \cap S_v == Ø,
    3. C_t \cap C_v != Ø and S_t \cap S_v != Ø,
    4. C_t \cap C_v != Ø and S_t \cap S_v == Ø,
    
    Variants where C_t \cap C_v != Ø (same for S_x):
    1. C_t == C_v
    2. |C_t \cap C_v| < |C_t \cup C_v|
    
    Restrictions: 
    Retriction of a set. eg. s_1 \in S_v and |S_v|=1, {'S_v':{'content:[s_1],'max_len',1}}
    """
    C_t,C_v,S_t,S_v=map(set,[[]]*4)
    restrictions = {**{'C_t':{},'C_v':{},'S_t':{},'S_v':{}},**restrictions}
    
    def filter_restrictions(C_t,C_v,S_t,S_v):
        for _set,_inv_set,k in zip([C_t,C_v,S_t,S_v],[C_v,C_t,S_v,S_t],['C_t','C_v','S_t','S_v']):
            if k in restrictions:
                if 'content' in restrictions[k]:
                    _set |= restrictions[k]['content']
                if 'not content' in restrictions[k]:
                    _set -= restrictions[k]['not content']
                
                if 'max_len' in restrictions[k]:
                    while restrictions[k]['max_len'] < len(_set):
                        entity = choice(list(_set))
                        if not ('content' in restrictions[k] and entity in restrictions[k]['content']):
                            _set.remove(entity)
                    
        return C_t,C_v,S_t,S_v
    
    def check_restrictions(C_t,C_v,S_t,S_v):
        for _set,k,inv_k in zip([C_t,C_v,S_t,S_v],['C_t','C_v','S_t','S_v'],['C_v','C_t','S_v','S_t']):
            if k in restrictions:
                if 'content' in restrictions[k] and 'not content' in restrictions[k]:
                    try:
                        assert len(restrictions[k]['content'].intersection(restrictions[k]['not content'])) < 1
                    except AssertionError:
                        raise AssertionError('Set %s content conflict.' % k)
                
                if 'content' in restrictions[k] and 'max_len' in restrictions[k]:
                    try:
                        assert len(restrictions[k]['content']) <= restrictions[k]['max_len']
                    except AssertionError:
                        raise AssertionError('Set %s content is longer than max length' % k)
                    if ((method == 1 and 'C' in k) or (method == 4 and 'S' in k) or method == 2) and 'content' in restrictions[inv_k]:
                        try:
                            assert restrictions[k]['content'].intersection(restrictions[inv_k]['content']) == set()
                        except AssertionError:
                            raise AssertionError('Intersection in %s content is not allowed in method %s.' % ('chemical' if method==1 else 'species',str(method)))
                    if method == 3 and 'content' in restrictions[inv_k]:
                        try:
                            assert restrictions[k]['content'].intersection(restrictions[inv_k]['content']) == set()
                        except AssertionError:
                            raise AssertionError('Intersection in set content is not allowed in method 3.')
                
    C,S = map(set,zip(*X))
    if method == 1:
        C_t,C_v = train_test_split(list(C),test_size=prop)
        if variant == 1:
            S_t,S_v = S, S
        else:
            S_t = choices(list(S),k=int((1-prop)*len(S)))
            S_v = choices(list(S),k=int(prop*len(S)))
            
    if method == 2:
        S_t,S_v = train_test_split(list(S),test_size=prop)
        C_t,C_v = train_test_split(list(C),test_size=prop)
    
    if method == 3:
        X_t, X_v = train_test_split(X,test_size=prop)
        C_t,S_t = map(set,zip(*X_t))
        C_v,S_v = map(set,zip(*X_v))
    
    if method == 4:
        S_t,S_v = train_test_split(list(S),test_size=prop)
        if variant == 1:
            C_t,C_v = C, C
        else:
            C_t = choices(list(C),k=int((1-prop)*len(C)))
            C_v = choices(list(C),k=int(prop*len(C)))
    
    C_t,C_v,S_t,S_v = map(set,[C_t,C_v,S_t,S_v])
    C_t,C_v,S_t,S_v = filter_restrictions(C_t,C_v,S_t,S_v)
    
    if method == 1: C_t -= C_v
    if method == 2:
        C_t -= C_v
        S_t -= S_v
    if method == 4: S_t -= S_v
    
    if method == 1:
        assert C_t.intersection(C_v) == set()
        if variant == 1:
            S_t = S_v
            assert S_t == S_v
        else:
            assert len(S_t.intersection(S_v)) < len(S_t.union(S_v))
            
    if method == 2:
        assert C_t.intersection(C_v) == set() and S_t.intersection(S_v) == set()
    
    if method == 3:
        assert len(C_t.intersection(C_v)) > 0 and len(S_t.intersection(S_v)) > 0
    
    if method == 4:
        assert S_t.intersection(S_v) == set()
        if variant == 1:
            C_t = C_v
            assert C_t == C_v
        else:
            assert len(C_t.intersection(C_v)) < len(C_t.union(C_v))
            
    check_restrictions(C_t,C_v,S_t,S_v)
            
    Xtr = []
    Xte = []
    ytr = []
    yte = []
    for x,y in zip(X,Y):
        c,s = x
        if c in C_t and s in S_t:
            Xtr.append(x)
            ytr.append(y)
        if c in C_v and s in S_v:
            Xte.append(x)
            yte.append(y)
            
    return Xtr,Xte,ytr,yte

class FilterFingerprints:
    def __init__(self):
        pass
    
    def fit(self,X):
        idx = []
        for i,a in enumerate(X.T):
            if len(np.unique(a)) > 1:
                idx.append(i)
        self.idx = idx
    def transform(self,X):
        if len(X.shape) > 1:
            return X[:,self.idx]
        else:
            return X[self.idx]
        
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)


def compile_model(model):
    model.compile(optimizer='adagrad',loss='log_cosh',metrics=['mae','mse',R2(name='r2')])

import math
def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def combine(Xs):
    n = map(len,Xs)
    l = max(*map(lambda x: lcm(len(x[0]),len(x[1])),product(Xs,Xs)))
    r = [l//a for a in n]
    tmp = []
    for X,a in zip(Xs,r):
        tmp.append(np.repeat(X,a,axis=0))
    return np.concatenate(tmp,axis=1)

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)

def run_model(C_t,C_v,S_t,S_v,y,
              experimental_features,
              fingerprints,
              chemical_embedding,
              species_embedding,
              chemical_features,
              merge_species=False):
    
    """
    Take four classes of chemicals, two pairs of siblings, test these on one-two species, combine siblings, combine cusins, see performance drop. Repeat on species side.
    Repeat with embeddings for chemicals and species and see the same performance on lower levels, but imporved over baseline on higher levels. 
    """
    
    """
    5-fold validation 
    + 1-fold test set
    """
    
    keys = set(y.keys())
    keys_t = keys.intersection(set(product(C_t,S_t)))
    keys_v = keys.intersection(set(product(C_v,S_v)))
    
    ytr,yte = map(lambda x:np.asarray([y[i] for i in x]),[keys_t,keys_v])
    if len(yte) < 1 or len(ytr) < 1:
        return None,None,None
    
    fingerprints_train,fingerprints_test = map(lambda x:np.asarray([fingerprints[i] for i,_ in x]),[keys_t,keys_v])
    chemical_embedding_train,chemical_embedding_test = map(lambda x:np.asarray([chemical_embedding[i] for i,_ in x]),[keys_t,keys_v])
    chemical_features_train,chemical_features_test = map(lambda x:np.asarray([chemical_features[i] for i,_ in x]),[keys_t,keys_v])
    
    species_embedding_train,species_embedding_test = map(lambda x:np.asarray([species_embedding[i] for _,i in x]),[keys_t,keys_v])
    experimental_features_train,experimental_features_test = map(lambda x:np.asarray([experimental_features[i] for i in x]),[keys_t,keys_v])
    
    species_one_hot_encoder = OneHotEncoder(sparse=False)
    sp_t = set(list(zip(*keys_t))[1])
    sp_v = set(list(zip(*keys_v))[1])
    sp = np.asarray(list(sp_t|sp_v)).reshape((-1,1))
    species_one_hot_encoder.fit(sp)
    species_one_hot_train,species_one_hot_test = map(lambda x:species_one_hot_encoder.transform(np.asarray(list(zip(*x))[1]).reshape((-1,1))),[keys_t,keys_v])
    
    
    if merge_species:
        for array in [species_embedding_train,species_one_hot_train,ytr]:
            for elem,loc in list_duplicates([c for c,_ in keys_t]): #i.e. mean where c is the same
                array[loc] = np.mean(array[loc])
        for array in [species_embedding_test,species_one_hot_test,yte]:
            for elem,loc in list_duplicates([c for c,_ in keys_v]):
                array[loc] = np.mean(array[loc])
        
    n_tr = ytr.shape[1]
    n_te = yte.shape[1]
    
    train_1 = combine([fingerprints_train,chemical_features_train,species_one_hot_train,experimental_features_train,ytr])
    train_2 = combine([fingerprints_train,chemical_features_train,species_embedding_train,chemical_embedding_train,experimental_features_train,ytr])
    test_1 = combine([fingerprints_test,chemical_features_test,species_one_hot_test,experimental_features_test,yte])
    test_2 = combine([fingerprints_test,chemical_features_test,species_embedding_test,chemical_embedding_test,experimental_features_test,yte])

    Xtr_1,ytr = train_1[:,:-n_tr],train_1[:,-n_tr:]
    Xtr_2,ytr = train_2[:,:-n_tr],train_2[:,-n_tr:]
    Xte_1,yte = test_1[:,:-n_te],test_1[:,-n_te:]
    Xte_2,yte = test_2[:,:-n_te],test_2[:,-n_te:]
    
    res1 = np.zeros(yte.ravel().shape)
    res2 = np.zeros(yte.ravel().shape)
    
    params = {'n_neighbors':[2,5,10,25,50,100],
              'weights':['uniform','distance']}
    
    n = min(len(ytr),5)
    
    FOLDS = 10
    for Xtr,Xte,res in zip([Xtr_1,Xtr_2],[Xte_1,Xte_2],[res1,res2]):
        for _ in range(FOLDS):
            regr = AdaBoostRegressor(n_estimators=10,loss='square')
            regr.fit(Xtr,ytr.ravel())
            res += regr.predict(Xte)/FOLDS
        
    return res1,res2,yte

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)

def get_species_name(ncbi_id):
    q = """
    select ?label where {
    ?s wdt:P685 "%s" ;
        wdt:P225 ?label .
    }
    """ % ncbi_id
    
    sparql.setQuery(q)
    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            out = result["label"]["value"]
        return out
    except:
        return ncbi_id

def encode_fingerprints(fingerprints_all):
    fingerprint_encoder, fingerprint_ae = create_auto_encoder(input_size=len(fingerprints_all[0]),dense_layers=(128,),noise=0.1)
    fingerprint_ae.compile(optimizer='adagrad',loss='binary_crossentropy')
    fingerprint_ae.fit(fingerprints_all,fingerprints_all,
            epochs=MAX_ENCODER_EPOCHS,
            callbacks=[EarlyStopping('loss',min_delta=1e-5)],
            verbose=0)
        
    return fingerprint_encoder.predict(fingerprints_all)
    
from sklearn.cluster import KMeans

# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        
        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
        sse.append(curr_sse)
    return sse
    
def define_chemical_clusters(fingerprints,k=15,use_pca=True):
    if not isinstance(fingerprints,list):
        fingerprints = [fingerprints]
    keys = set.intersection(*[set(f.keys()) for f in fingerprints])
    array =  np.concatenate([np.asarray([v[k] for k in keys]) for v in fingerprints],axis=1)
    
    if use_pca:
        array = PCA(2).fit_transform(array)
    if k < 0:
        sse = calculate_WSS(array,25)
        k = np.argmin(sse) + 1
        plt.plot(sse)
        plt.show()
    clusters = defaultdict(set)
    kmeans = KMeans(n_clusters = k).fit(array)
    cp = kmeans.predict(array)
    for k,v in zip(keys,cp):
        clusters[v].add(k)
    return clusters, kmeans.cluster_centers_

def merge_closest(clusters,cluster_centers,ord=2):
    dist = {}
    for i,cc1 in enumerate(cluster_centers):
        for j,cc2 in enumerate(cluster_centers):
            if i == j: continue
            dist[(i,j)] = np.linalg.norm(cc1-cc2,ord=ord)
    
    if len(dist) > 1:
        merge,_ = sorted(dist.items(),key=lambda x:x[1])[0]
    else:
        merge = (i,j)
    
    k1,k2 = merge
    cluster_centers[k1] = np.mean([cluster_centers[k1],cluster_centers[k2]],axis=0)
    cluster_centers = np.delete(cluster_centers,k2,axis=0)
    
    clusters[k1] |= clusters[k2]
    clusters.pop(k2,None)
    return clusters, cluster_centers
    
    
def filter_data(X,Y,C_t,C_v,S_t,S_v):
    
    Xtr,Xte,ytr,yte = [],[],[],[]
    for x,y in zip(X,Y):
        c,s = x
        if c in C_t and s in S_t:
            Xtr.append(x)
            ytr.append(y)
        if c in C_v and s in S_v:
            Xte.append(x)
            yte.append(y)
    return Xtr,Xte,ytr,yte

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/media/erik/Mass/Dropbox/NIVA_GITLAB/pySMIfp')

from smiles_fingerprints import smiles_fingerprint

def load_smiles_fingerprints():
    q = """
    select ?chembl ?smiles where {
        ?c wdt:P233 ?smiles ;
            wdt:P592 ?chembl .
        }
    """ 
    converter = {}
    sparql.setQuery(q)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        ch = result["chembl"]["value"]
        smi = result['smiles']['value']
        smifp = smiles_fingerprint(smi)
        converter['http://rdf.ebi.ac.uk/resource/chembl/molecule/'+ch] = smifp
    
    return converter

def save_smiles_fingerprints(fp,filename='data/smiles_fingerprints.csv'):
    a = {}
    for i in range(len(smiles_fingerprint('C'))):
        a['sig%s'%str(i)] = [array[i] for _,array in fp.items()]
    
    df = pd.DataFrame(data={'chemical':list(fp.keys()),**a})
    df.to_csv(filename)
    
    
def read_smiles_fingerprints(filename):
    df = pd.read_csv(filename)
    cols = [c for c in df.columns if 'sig' in c]
    chemicals = df['chemical'].values
    arrays = df[cols].values
    
    return dict(zip(chemicals,np.asarray(arrays)))
    
def chemical_similarities(fingerprints):
    keys = fingerprints.keys()
    array = np.asarray([i for k,i in fingerprints.items()])
    
    sim = []
    for a in array:
        v = a @ array.T
        w = np.sum(a) + np.sum(array,axis=1)
        sim_score = 2*v/w
        sim.append(sim_score)
    
    return {k:s for k,s in zip(keys,sim)}
    
def main():
    """
    organic = obo['CHEBI_50860']
    inorganic = obo['CHEBI_24835']
    """
    
    
    model = 'ComplEx'
    g1_parts = [[0],[0,1],[0,1,2]]
    g2_parts = [[0],[0,1]]
    p = list(product(g1_parts,g2_parts))
    p += [p[-1]]
    ul = (False,False)
    f1,f2=[],[]
    for g1p,g2p,in p:
        
        for lit,gp,fs,name in zip([*ul],[g1p,g2p],[f1,f2],['_chemical_','_taxonomy_']):
            fs.append(model+name+str(hash((lit,*gp))))
            
        if (g1p,g2p) == p[-1]:
            ul = (True,True)
    
    
    organic_chemicals = set()
    inorganic_chemicals = set()
    salts = set()
    for i in range(1,10):
        df = pd.read_csv('./data/chemical_group_%s.csv' % str(i),index_col='parent')
        try:
            organic_chemicals |= set(df.loc['http://purl.obolibrary.org/obo/CHEBI_50860','children'].split(','))
        except:
            pass
        try:
            inorganic_chemicals |= set(df.loc['http://purl.obolibrary.org/obo/CHEBI_24835','children'].split(','))
        except:
            pass
        try:
            salts |= set(df.loc['http://purl.obolibrary.org/obo/CHEBI_24866','children'].split(','))
        except:
            pass
        
    print('Num organic chemicals',len(organic_chemicals))
    print('Num inorganic chemicals',len(inorganic_chemicals))
    print('Num salts',len(salts))
    
    C = organic_chemicals
    try:
        smiles_fingerprints = read_smiles_fingerprints('./data/smiles_fingerprints.csv')
    except FileNotFoundError:
        smiles_fingerprints = load_smiles_fingerprints()
        save_smiles_fingerprints(smiles_fingerprints,'./data/smiles_fingerprints.csv')
    
    mms = MinMaxScaler().fit_transform(np.asarray([smiles_fingerprints[k] for k in smiles_fingerprints]))
    smiles_fingerprints = dict(zip(smiles_fingerprints,mms))
    
    X,Y,experimental_features = load_data('./data/experiments.csv',filter_chemicals=None, filter_species=None)
    pubchem_fingerprints = load_fingerprints('./data/chemicals_fingerprints.csv')
    
    Y = {k:y for k,y in zip(X,Y)}
    
    pubchem_fingerprints = chemical_similarities(pubchem_fingerprints)
    
    chemical_embedding = load_embeddings('./data/embeddings/%s_entity_embeddings.npy' % f1[0],
                                               './data/embeddings/%s_entity_ids.npy' % f1[0])
    species_embedding = load_embeddings('./data/embeddings/%s_entity_embeddings.npy' % f2[0],
                                               './data/embeddings/%s_entity_ids.npy' % f2[0])
    chemical_features = load_features('./data/chemicals_features.csv')
    chemical_features = dict(zip(chemical_features,MinMaxScaler().fit_transform(np.asarray([chemical_features[k] for k in chemical_features]))))
    
    for cf in [QuantileTransformer(n_quantiles=100,output_distribution='normal')]:
        chemical_embedding = dict(zip(chemical_embedding,cf.fit_transform(np.asarray([chemical_embedding[k] for k in chemical_embedding]))))
        
    for cf in [QuantileTransformer(n_quantiles=100,output_distribution='normal')]:
        species_embedding = dict(zip(species_embedding,cf.fit_transform(np.asarray([species_embedding[k] for k in species_embedding]))))
    
    
    species_divisions = defaultdict(set)
    for k in range(1,2):
        df = pd.read_csv('./data/species_groups_%s.csv' % str(k), index_col='parent')
        for s in df.index:
            species_divisions[s] |= set(df.loc[s,'children'].split(','))
    species_divisions = dict(filter(lambda x:len(x[1])>5,species_divisions.items()))
    #for k in species_divisions:
        #print(get_species_name(k.split('/')[-1]))
    
    #species_divisions = defaultdict(set)
    #df = pd.read_csv('./data/species_divisions.csv', index_col='parent')
    #for s in df.index:
        #species_divisions[s] |= set(df.loc[s,'children'].split(','))
    
    C = set.intersection(*map(lambda k:set(k.keys()),[smiles_fingerprints,pubchem_fingerprints,chemical_features,chemical_embedding]))
    for d in [smiles_fingerprints,pubchem_fingerprints,chemical_embedding,chemical_features]:
        for c in set(d.keys()):
            if not c in C:
                d.pop(c,None)
    
    n = 7
    clusters, cluster_centers = define_chemical_clusters([smiles_fingerprints],k=max(-1,n),use_pca=False)
    
    print(*map(lambda x:len(x[1]),clusters.items()))
    
    data = {}
    
    all_runs = {}
    TOP_K = 10
    
    while True:
    
        for C,S in tqdm(product(clusters,species_divisions),total=len(clusters)*len(species_divisions)):
            k = [C,S]
            C = list(clusters[C])
            S = species_divisions[S]
            
            k[1] = get_species_name(k[1].split('/')[-1])
            
            loo = LeaveOneOut()
            predictions = []
            y_true = []
            for train_index, test_index in loo.split(C):
                C_t = [C[i] for i in train_index]
                C_v = [C[i] for i in test_index]
            
                r1,r2,yte = run_model(C_t,C_v,S,S,Y,
                                experimental_features,
                                pubchem_fingerprints,
                                chemical_embedding,
                                species_embedding,
                                chemical_features,
                                merge_species=True)
                
                if r1 is None and r2 is None: continue
                r1 = np.mean(r1)
                r2 = np.mean(r2)
                y_true.append(np.mean(yte))
                predictions.append((r1,r2))
            y_true, predictions = map(np.asarray,[y_true,predictions])
            if len(predictions) < 10: continue
            try:
                if len(predictions.shape) < 2:
                    predictions = np.expand_dims(predictions,axis=1)
                rsq_1 = r2_score(y_true,predictions[:,0])
                rsq_2 = r2_score(y_true,predictions[:,1])
                all_runs[tuple(k)] = (rsq_1,rsq_2)
            except ValueError:
                pass
            
        all_runs = dict(sorted(all_runs.items(),key=lambda x: sum(x[1])/2,reverse=True))
        print(all_runs)
        
        data[len(cluster_centers)] = all_runs
        
        if len(cluster_centers) > 0:
            clusters, cluster_centers = merge_closest(clusters,cluster_centers)
            for k in list(all_runs.keys())[:TOP_K]:
                _,s = k
                species_divisions.pop(k,None)
        else:
            break
    
    pd.to_pickle(data,'chemical_cluster_merging.pkl')
    
    exit()
    
    ks = set()

    for k in species_divisions:
        S = species_divisions[k]
        still_true = True
        for k_c in clusters:
            C = clusters[k_c]
            Xtr,Xte,ytr,yte = filter_data(X,Y,C,C,S,S)
            if count(Xtr,Xte) > 100: ks.add(k)
    
    for k in tqdm(ks):
        
        n=6
        clusters, cluster_centers = define_chemical_clusters([smiles_fingerprints],k=max(-1,n))
       
        S = species_divisions[k]
        sn = get_species_name(k.split('/')[-1])
        
        results = defaultdict(list)
        i = 0
        while True:
            k_c = sorted(clusters,key=lambda x:len(clusters[x]),reverse=True)[0]
            C_t = clusters[k_c]
            if len(C_t) < 1: continue
            C_t,C_v = train_test_split(list(C_t),test_size=0.25)
        
            S_t = S
            S_v = S
            Xtr,Xte,ytr,yte = filter_data(X,Y,C_t,C_v,S_t,S_v)
            
            try:
                assert count(Xtr,Xte) > 20
                r1,r2 = run_model(Xtr,
                        Xte,
                        ytr,
                        yte,
                        experimental_features,
                        pubchem_fingerprints,
                        chemical_embedding,
                        species_embedding,
                        chemical_features,
                        merge_species=True)
            except AssertionError:
                r1,r2 = float('nan'), float('nan')
            except np.AxisError:
                r1,r2 = float('nan'), float('nan')
                
            results[i].append((r1,r2))
                
            clusters, cluster_centers = merge_closest(clusters,cluster_centers)
            if len(cluster_centers) < 1:
                break
            i += 1
            
        v0 = [[v[0] for v in results[k]] for k in results]
        v1 = [[v[1] for v in results[k]] for k in results]
        
        fig, ax = plt.subplots()
        
        for x,color,ran in zip([v0,v1],['red','green'],[np.arange(0,len(v0)*2,2),np.arange(1,len(v1)*2,2)]):
            
            mins = [np.nanmin(a) for a in x]
            maxes = [np.nanmax(a) for a in x]
            means = [np.nanmean(a) for a in x]
            std = [np.nanstd(a) for a in x]
            
            mins,maxes,means,std = map(np.asarray,[mins,maxes,means,std])
            
            ax.bar(ran,maxes,width=0.5,color=color)
            
            #plt.ylim(-1,1)
        ax.set_xticks(np.arange(0.5,len(v0)*2,2))
        ax.set_xticklabels(('%s Clusters' % str(abs(i)) for i in range(-n,0)))
        plt.savefig('./plots/chemical_clusters_taxon_%s.png' % sn)
        
    exit()
    
    #def tqdm(x,**params):
        #return x
    
    for filter_chemicals,string,TOP_K in tqdm(zip([inorganic_chemicals | salts],['organic'],[4]),total=1,desc='Chemical Groups'):
        
        #if string=='organic': continue
        
        for division in tqdm(S_v,total=len(S_v),desc='Divisions'):
            if not len(S_v[division]) > 1: continue
            
            model_params={'encode':False,'train_ae_fingerprints':False,'train_ae_species':False}
            results = [[]]*TOP_K
            f = lambda _s: sum([1 for c,s in X if (s == _s and c in C-filter_chemicals)])
            tmp_division = list(sorted(S_v[division],key=f,reverse=True))[:TOP_K]
            for i,s_v in tqdm(enumerate(tmp_division),desc='Species in division %s' % division,leave=False,total=len(tmp_division)):
                
                C_restriction = {'C_v':{'not content':filter_chemicals},'C_t':{'not content':filter_chemicals}}
                configs = []
                #Method 1
                configs.append((1, 1, {'S_v':{'content':set([s_v]),'max_len':1}}))
                configs.append((1, 2, {'S_v':{'content':set([s_v]),'max_len':1}}))
                
                #Method 2
                configs.append((2, 1, {'S_v':{'content':set([s_v]),'max_len':1}}))
                
                #Method 3
                configs.append((3, 1, {'S_v':{'content':set([s_v]),'max_len':1}}))
                configs.append((3, 2, {'S_v':{'content':set([s_v]),'max_len':1}}))
                
                #Method 4
                configs.append((4, 1, {'S_v':{'content':set([s_v]),'max_len':1}}))
                configs.append((4, 2, {'S_v':{'content':set([s_v]),'max_len':1}}))
                
                tmp_res = np.zeros((len(configs),2))
                
                for j,config in tqdm(enumerate(configs),total=len(configs),leave=False,desc='Configs'):
                    m,v,res = config
                    r1_tmp = []
                    r2_tmp = []
                    for _ in range(10):
                        tf.keras.backend.clear_session()
                        prop = 0.3
                        Xtr,Xte,ytr,yte = data_split(X,Y,restrictions={**res,**C_restriction},method=m,variant=v,prop=prop)
                    
                        try:
                            r1,r2 = run_model(Xtr,
                                    Xte,
                                    ytr,
                                    yte,
                                    experimental_features,
                                    fingerprints,
                                    chemical_embedding,
                                    species_embedding,
                                    model_params=model_params)
                        except:
                            r1,r2=0,0
                            
                        r1_tmp.append(r1)
                        r2_tmp.append(r2)
                    
                    tmp_res[j,0] = np.mean(r1_tmp)
                    tmp_res[j,1] = np.mean(r2_tmp)
                    
                results[i] = tmp_res
            
            fig, axs = plt.subplots(1,len(results),figsize=(40, 10))
            for i,ax in enumerate(axs):
                ms = results[i]
                baseline = ms[:,0]
                over = ms[:,1]
                
                baseline = np.nan_to_num(baseline, nan=0.0,posinf=0.0, neginf=0.0)
                over = np.nan_to_num(over, nan=0.0,posinf=0.0, neginf=0.0)
                     
                width = 0.4
                ax.bar(np.arange(0,len(baseline)*2,2),baseline,width,color='red')
                ax.bar(np.arange(1,len(baseline)*2,2),over,width,color='green')
                ax.set_title(get_species_name(tmp_division[i].split('/')[-1]))
                ax.set_xticks(np.arange(0.5,len(baseline)*2,2))
                ax.set_xticklabels((str(i) for i in range(len(configs))))
                ax.set_ylim(0,max(*over,*baseline)+0.1)
                
            plt.savefig('plots/division_%s_%s.png' % (division,string))
            
    
if __name__ == '__main__':
    main()
    



















