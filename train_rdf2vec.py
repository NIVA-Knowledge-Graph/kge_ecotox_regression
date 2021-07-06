


from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer

import pandas as pd
from rdflib import Graph, URIRef
import numpy as np
from main import load_data

import rdflib

d = './data/embeddings/'
    
pdf = [pd.read_csv('./data/chemicals_%s.csv' % str(i)) for i in range(3)]
    
kg1 = pd.concat(pdf)
    
kg2 = pd.read_csv('./data/taxonomy.csv') 

X,_ = load_data('./data/experiments.csv')

entities1 = list(set(map(rdflib.URIRef,list(zip(*X))[0])))
entities2 = list(set(map(rdflib.URIRef,list(zip(*X))[1])))

for kg,kg_name,entities in zip([kg1,kg2],['chemical','taxonomy'],[entities1,entities2]):
    g = Graph()
    for t in zip(kg['subject'],kg['predicate'],kg['object']):
        g.add(tuple(map(rdflib.URIRef,t)))
    g.serialize('tmp.ttl',format='ttl')

    kg = KG(location="tmp.ttl",file_type='ttl')
    
    walkers = [RandomWalker(4, 5, UniformSampler())]
    transformer = RDF2VecTransformer(walkers=walkers)
    
    embeddings = transformer.fit_transform(kg,entities)
    
    np.save(d + 'rdf2vec_%s_entity_embeddings.csv' % kg_name, embeddings)
    np.save(d + 'rdf2vec_%s_entity_ids.csv' % kg_name, np.asarray(list(enumerate(entities))))
