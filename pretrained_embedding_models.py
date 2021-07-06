
import sys
import os

from itertools import product
from KGEkeras import DistMult, HolE, TransE, HAKE, ConvE, ComplEx, ConvR, RotatE, pRotatE, ConvKB, CosinE

from kerastuner import RandomSearch, HyperParameters, Objective, Hyperband, BayesianOptimization

from random import choice
from collections import defaultdict

from tensorflow.keras.losses import binary_crossentropy,hinge,mean_squared_error
from tensorflow.keras import Input
from tensorflow.keras import Model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback, TerminateOnNaN, ReduceLROnPlateau
from sklearn.metrics.cluster import completeness_score

from tensorflow.keras.optimizers import Adam
import json

import tensorflow as tf

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from KGEkeras import loss_function_lookup
from lib.utils import generate_negative, oversample_data, load_data
from tqdm import tqdm

import string
import random

from random import choices
from lib.hptuner import HPTuner
import pickle

try:
    from tensorflow_addons.callbacks import TimeStopping
except:
    pass

            
from rdflib import Graph, URIRef, Literal, Namespace
from KGEkeras import LiteralConverter
from sklearn.decomposition import PCA


SECONDS_PER_TRAIL = 600
SECONDS_TO_TERMINATE = 3600
SEARCH_MAX_EPOCHS = 10
MAX_EPOCHS = 200
MIN_EPOCHS = 50
MAX_TRIALS = 20
PATIENCE = 10

EPSILON = 10e-7

models = {
            #'DistMult':DistMult,
            #'TransE':TransE,
            #'HolE':HolE,
            'ComplEx':ComplEx,
            #'HAKE':HAKE,
            #'pRotatE':pRotatE,
            #'RotatE':RotatE,
            #'ConvE':ConvE,
            #'ConvKB':ConvKB,
         }

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, kg, ns=10, batch_size=32, shuffle=True):
        self.batch_size = min(batch_size,len(kg))
        self.kg = kg
        self.ns = ns
        self.num_e = len(set([s for s,_,_ in kg])|set([o for _,_,o in kg]))
        self.shuffle = shuffle
        self.indices = list(range(len(kg)))
        
        self.on_epoch_end()

    def __len__(self):
        return len(self.kg) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        tmp_kg = np.asarray([self.kg[i] for i in batch])
        
        negative_kg = generate_negative(tmp_kg,N=self.num_e,negative=self.ns)
        X = oversample_data(kgs=[tmp_kg,negative_kg])
    
        return X, None 

def build_model(hp):
    
    params = hp.copy()
    params['e_dim'] = params['dim']
    params['r_dim'] = params['dim']
    params['name'] = 'embedding_model'
    
    embedding_model = models[params['embedding_model']]
    embedding_model = embedding_model(**params)
    triple = Input((3,))
    ftriple = Input((3,))
    
    inputs = [triple, ftriple]
    
    score = embedding_model(triple)
    fscore = embedding_model(ftriple)
    
    loss_function = loss_function_lookup(params['loss_function'])
    loss = loss_function(score,fscore,params['margin'] or 1, 1)
    
    model = Model(inputs=inputs, outputs=loss)
    model.add_loss(loss)
    
    model.compile(optimizer=Adam(learning_rate=ExponentialDecay(params['learning_rate'],decay_steps=100000,decay_rate=0.96)),
                  loss=None)
    
    return model


def optimize_model(model, kg, lit=False, name='name', hp=None):
    
    if lit:
        lc = LiteralConverter(kg)
        literals = lc.fit_transform()
        kg = lc.g
        literals = PCA(min(len(literals[0]),100)).fit_transform(literals)
    else:
        literals = None
    
    kg -= [(s,p,o) for s,p,o in kg if isinstance(o,Literal)]
    
    entities = set(kg.subjects()) | set(kg.objects())
    relations = set(kg.predicates())
    me = {k:i for i,k in enumerate(entities)}
    mr = {k:i for i,k in enumerate(relations)}
    kg = list(map(lambda x: (me[x[0]],mr[x[1]],me[x[2]]), kg))
   
    bs = 512
    
    kg = np.asarray(kg)
    
    model_name = model
    
    N = len(me)
    M = len(mr)
        
    hptuner = HPTuner(runs=MAX_TRIALS, objectiv_direction='min')
    hptuner.add_value_hp('gamma',0,21)
    hptuner.add_value_hp('dim',100,401,dtype=int)
    hptuner.add_value_hp('negative_samples',10,101,dtype=int)
    hptuner.add_value_hp('margin',1,11,dtype=int)
    hptuner.add_list_hp('loss_function',['pairwize_hinge','pairwize_logistic','pointwize_hinge','pointwize_logistic'],exhaustive=True)
    
    hptuner.add_fixed_hp('embedding_model',model)
    hptuner.add_fixed_hp('dp',0.2)
    hptuner.add_fixed_hp('hidden_dp',0.2)
    hptuner.add_fixed_hp('num_entities',N)
    hptuner.add_fixed_hp('num_relations',M)
    
    if hp:
        for k,i in hp.items():
            hptuner.add_fixed_hp(k,i)
            
    hptuner.add_fixed_hp('num_entities',N)
    hptuner.add_fixed_hp('num_relations',M)
    
    hptuner.add_fixed_hp('learning_rate',0.001)
    hptuner.add_fixed_hp('regularization',0.001)
    if lit:
        hptuner.add_fixed_hp('literals',literals)
        hptuner.add_fixed_hp('literal_activation','tanh')
    
    if hp:
        hptuner.next_hp_config()
        hptuner.add_result(0.0)
    
    with tqdm(total=hptuner.runs, desc='Trials') as pbar:
        while hptuner.is_active and hp is None:
            hp = hptuner.next_hp_config()
            model = build_model(hp)
            tr_gen = DataGenerator(kg, batch_size=bs, shuffle=True, ns=hp['negative_samples'])
            hist = model.fit(tr_gen,epochs=SEARCH_MAX_EPOCHS,verbose=2, callbacks=[EarlyStopping('loss'),TerminateOnNaN()])
            score = hist.history['loss'][-1]/hist.history['loss'][0]
            hptuner.add_result(score)
            tf.keras.backend.clear_session()
            pbar.update(1)
        
    hp = hptuner.best_config()
    
    #if hp is None:
        #with open('./pretrained_hp/%s%s_kg.json' % (model_name,name), 'w') as fp:
            #json.dump(hp, fp)
    
    model = build_model(hp)
    tr_gen = DataGenerator(kg, batch_size=bs, shuffle=True, ns=hp['negative_samples'])
    hist = model.fit(tr_gen,epochs=MAX_EPOCHS, verbose=2, callbacks=[EarlyStopping('loss',patience=PATIENCE), TerminateOnNaN()])
    if np.isnan(hist.history['loss'][-1]):
        print(model_name,'nan loss.')
        return optimize_model(model_name,kg,lit,name,None)
    
    for l in model.layers:
        if isinstance(l,models[model_name]):
            m = l.name
           
    m, W1, W2 = model, model.get_layer(m).entity_embedding.get_weights()[0], model.get_layer(m).relational_embedding.get_weights()[0]
           
    m.save_weights('pretrained_models/model/'+name)
    np.save(name+'_entity_embeddings.npy', W1)
    np.save(name+'_entity_ids.npy',np.asarray(list(zip(entities,range(len(entities))))))
    np.save(name+'_relational_embeddings.npy', W2)
    np.save(name+'_relation_ids.npy',np.asarray(list(zip(relations,range(len(relations))))))
    
def main():
    d = './data/embeddings/'
    
    use_literals = product([False,True],[False,True])
    g1_parts = [[0],[0,1],[0,1,2]]
    g2_parts = [[0],[0,1]]
    p = list(product(g1_parts,g2_parts))
    p += [p[-1]]
    ul = (False,False)
    for g1p,g2p in tqdm(p):
        g1,g2 = Graph(),Graph()
        for i in g1p:
            g = Graph()
            g.load('./data/chemicals_%s.ttl' % str(i),format='ttl')
            g1 += g
        for i in g2p:
            g = Graph()
            g.load('./data/taxonomy_%s.ttl' % str(i),format='ttl')
            g2 += g
        
        for lit,gp,kg,name in zip([*ul],[g1p,g2p],[g1,g2],['_chemical_','_taxonomy_']):
            #hp_file = '../KGE-CEP/pretrained_hp/%s%s_kg.json' % (model,name)
            hp = {'e_dim':100,
                  'negative_samples':10,
                  'loss_function':'pairwize_logistic'}
            
            model = 'ComplEx'
            f = d+model+name+str(hash((lit,*gp)))
            optimize_model(model,kg,lit,name=f,hp=hp)
            
            tf.keras.backend.clear_session()
            
        if (g1p,g2p) == p[-1]:
            ul = (True,True)
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
