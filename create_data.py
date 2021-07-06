

"""
TODO:
    - Load LC50 data from ECOTOX. 
    - Take median per chemical species pairs. 
    - Defined chemical groups. 
    - Export files per chemical groups and each species. 
    
    - Forall chemicals and species export relevant KGs. 
"""

from tera.DataAggregation import Taxonomy, Effects, Traits
from tera.DataAccess import EffectsAPI
from tera.DataIntegration import DownloadedWikidata, LogMapMapping
from tera.utils import strip_namespace, unit_conversion
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal, BNode, Namespace
from rdflib.namespace import RDFS, RDF

cco = Namespace('http://rdf.ebi.ac.uk/terms/chembl#')
skos = Namespace('http://www.w3.org/2004/02/skos/core#')
obo = Namespace('http://purl.obolibrary.org/obo/')

import pandas as pd
from collections import defaultdict

import pubchempy as pcp
import numpy as np

def get_subgraph(to_visit, graph, backtracking=0):
    out = Graph()
    visited = set()
    while to_visit:
        curr = to_visit.pop()
        visited.add(curr)
        tmp = set(graph.triples((curr,None,None)))
        for t in tmp: 
            out.add(t)
        to_visit |= set([o for _,_,o in tmp if not isinstance(o,Literal)])
        to_visit -= visited
        
    if backtracking > 0:
        tmp = set()
        for s in set([s for s,_,_ in out]):
            tmp |= set(graph.subjects(object=s))
        for t in out:
            graph.remove(t)
        return out + get_subgraph(tmp, graph, backtracking-1)
        
    return out


def load_endpoint_data():

    ed = Effects(directory='../ecotox_data/',verbose=False)
    
    species_mapping = LogMapMapping(filename='./data/final_mappings.txt')
    chemicals_mappings = DownloadedWikidata(filename='./data/cas_to_chembl.csv')
    species_mapping.load()
    chemicals_mappings.load()
    
    ncbi_namespace = Namespace('https://www.ncbi.nlm.nih.gov/taxonomy/')
    species_mapping = [(ed.namespace['taxon/'+k],ncbi_namespace['taxon/'+i.pop(0)]) for k,i in species_mapping.mappings.items()]
    ed.replace(species_mapping)
    
    chembl_namespace = Namespace('http://rdf.ebi.ac.uk/resource/chembl/molecule/')
    chemicals_mappings = [(ed.namespace['cas/'+k],chembl_namespace[i.pop(0)]) for k,i in chemicals_mappings.mappings.items()]
    ed.replace(chemicals_mappings)
    
    endpoints = EffectsAPI(dataobject=ed, verbose=True).get_endpoint(c=None, s=None)

    d = defaultdict(list)
    for c,s,cc,cu,ep,ef,sd,sdu in endpoints:
        try:
            sd = float(sd)
        except: 
            continue
        if 'day' in str(sdu).lower():
            sd *= 24
        elif 'week' in str(sdu).lower():
            sd *= (7*24)
        elif 'hour' in str(sdu).lower():
            sd *= 1
        else:
            continue
        
        if ('LC50' in str(ep) or 'LD50' in str(ep) or ('EC50' in str(ep) and 'MOR' in str(ef))) and ('ncbi' in str(s) and 'chembl' in str(c)):
            try:
                factor = unit_conversion(str(cu),'http://qudt.org/vocab/unit#MilligramPerLitre')
            except:
                factor = 0
                
            if factor > 0:
                cc = float(cc)
                cc = cc*factor
                    
                d['chemical'].append(str(c))
                d['species'].append(str(s))
                d['concentration'].append(cc)
                d['study_duration'].append(sd)
    
    df = pd.DataFrame(data=d)
    df.to_csv('./data/experiments.csv')

def fingerprints():
    df = pd.read_csv('./data/experiments.csv')
    mapping = DownloadedWikidata(filename='./data/chembl_to_cid.csv')
    
    to_look_for = mapping.convert(set(df['chemical']),reverse=False,strip=True)
    to_look_for = set([URIRef('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID'+str(i)) for k,i in to_look_for.items() if i != 'no mapping'])
    
    out = []
    fp = []
    for c,c2 in tqdm(zip(to_look_for,set(df['chemical'])),total=len(to_look_for)):
        try:
            compound = pcp.Compound.from_cid(int(c.split('CID')[-1]))
            fp.append(bin(int(compound.fingerprint,16))[2:])
            out.append(c2)
        except:
            pass
        
    df = pd.DataFrame(data={'chemical':out,'fingerprint':fp})
    df.to_csv('./data/chemicals_fingerprints.csv')

def chemical_features():

    df = pd.read_csv('./data/experiments.csv')
    mapping = DownloadedWikidata(filename='./data/chembl_to_cid.csv')
    
    to_look_for = mapping.convert(set(df['chemical']),reverse=False,strip=True)
    to_look_for = set([URIRef('http://rdf.ncbi.nlm.nih.gov/pubchem/compound/CID'+str(i)) for k,i in to_look_for.items() if i != 'no mapping'])
    
    out = defaultdict(list)
    fp = []
    fs = ['xlogp','exact_mass','tpsa','complexity','charge']
    for c,c2 in tqdm(zip(to_look_for,set(df['chemical'])),total=len(to_look_for)):
        try:
            compound = pcp.Compound.from_cid(int(c.split('CID')[-1]))
            tmp = compound.to_dict(fs)
            for k in tmp:
                out[k].append(tmp[k])
            out['chemical'].append(c2)
        except:
            pass
        
    df = pd.DataFrame(data=out)
    df.to_csv('./data/chemicals_features.csv')


def load_chemical_groups():
    """
    Split (in)organic
    CHEBI:50860 (organic)
    CHEBI:24835 (inorganic)
    """
    
    df = pd.read_csv('./data/experiments.csv')
    
    chemicals = df['chemical']
    
    graph = Graph()
    graph.parse('../chembl/chembl_26.0_molecule_chebi_ls.ttl',format='ttl')
    mapping = defaultdict()
    for c in chemicals:
        c = URIRef(c)
        m = list(graph.objects(subject=c,predicate=skos['exactMatch']))
        if m:
            mapping[c]=m.pop(0)
    
    chebi_graph = Graph()
    chebi_graph.parse('../chebi/chebi.ttl',format='ttl')
    chebi_graph = replace(chebi_graph,{i:k for k,i in mapping.items()})
    
    for steps in range(1,20):
        out = defaultdict(set)
        desendents=set()
        for c in map(lambda x:URIRef(x), chemicals):
            p = ' / '.join('<'+r+'>' for r in [RDFS.subClassOf]*steps)
            qres = chebi_graph.query(
            """SELECT DISTINCT ?parent
                WHERE {
                    <%s> %s ?parent
                }""" % (str(c),p))
            
            for r in qres:
                desendents.add((c,r[-1]))
                
        for c,p in desendents:
            out[p].add(c)
        
        df = pd.DataFrame(data={'parent':list(out.keys()),'children':[','.join(out[k]) for k in out]})
        df.to_csv('./data/chemical_group_%s.csv' % str(steps))
        
def load_species_groups():
    df = pd.read_csv('./data/experiments.csv')
    t = Graph()
    t.load('./data/taxonomy_0.ttl',format='ttl')
    species = set(df['species'])
    
    for steps in range(0,20):
        out_taxon = defaultdict(set)
        out_division = defaultdict(set)
        desendents = set()
        for c in map(URIRef,species):
            p = ' / '.join('<'+r+'>' for r in [RDF.type,*[RDFS.subClassOf]*steps])
            qres = t.query(
            """SELECT DISTINCT ?parent
                WHERE {
                    <%s> %s ?parent
                }""" % (str(c),p))
            
            for r in qres:
                desendents.add((c,r[-1]))
        
        for c,p in desendents:
            if 'division' in str(p):
                out_division[p].add(c)
            else:
                out_taxon[p].add(c)
                    
        df = pd.DataFrame(data={'parent':list(out_division.keys()),'children':[','.join(out_division[k]) for k in out_division]})
        df.to_csv('./data/species_divisions.csv')
        
        df = pd.DataFrame(data={'parent':list(out_taxon.keys()),'children':[','.join(out_taxon[k]) for k in out_taxon]})
        df.to_csv('./data/species_groups_%s.csv' % str(steps))
        
def replace(graph,mapping):
    for s,p,o in graph:
        if s in mapping:
            graph.remove((s,p,o))
            graph.add((mapping[s],p,o))
        if o in mapping:
            graph.remove((s,p,o))
            graph.add((s,p,mapping[o]))
        if p in mapping:
            graph.remove((s,p,o))
            graph.add((s,mapping[p],o))
        
    return graph

def load_chemical_graph():
    df = pd.read_csv('./data/experiments.csv')
    
    mapping = DownloadedWikidata(filename='./data/chembl_to_mesh.csv')
    mapping.load()
    mapping = {URIRef('http://id.nlm.nih.gov/mesh/'+i.pop(0)):URIRef('http://rdf.ebi.ac.uk/resource/chembl/molecule/'+k) for k,i in mapping.mappings.items()}
    
    mesh_graph = Graph()
    mesh_graph.parse('../mesh/mesh.nt',format='nt')
    mesh_graph = replace(mesh_graph,mapping)
    
    graph = Graph()
    graph.parse('../chembl/chembl_26.0_molecule_chebi_ls.ttl',format='ttl')
    mapping = defaultdict()
    for c in df['chemical']:
        c = URIRef(c)
        m = list(graph.objects(subject=c,predicate=skos['exactMatch']))
        if m:
            mapping[c]=m.pop(0)
    
    chebi_graph = Graph()
    chebi_graph.parse('../chebi/chebi.ttl',format='ttl')
    chebi_graph = replace(chebi_graph,{i:k for k,i in mapping.items()})
    
    chembl_graph = Graph()
    for f in [#'../chembl/chembl_26.0_molecule.ttl',
              '../chembl/chembl_26.0_molhierarchy.ttl',
              '../chembl/chembl_26.0_target.ttl',
              '../chembl/chembl_26.0_targetrel.ttl',
              '../chembl/chembl_26.0_moa.ttl']:
        chembl_graph.parse(f,format='ttl')
    
    for i,g in enumerate([mesh_graph,chebi_graph,chembl_graph]):
        graph = get_subgraph(set([URIRef(a) for a in set(df['chemical'])]), g, backtracking=0)
        graph.serialize('./data/chemicals_%s.ttl' % str(i),format='ttl')
        
def load_taxonomy_graph():
    df = pd.read_csv('./data/experiments.csv')
    
    t = Taxonomy(directory='../taxdump/', verbose=True, taxon_namespace='http://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id=')
    ne = DownloadedWikidata(filename='./data/ncbi_to_eol.csv', verbose=False)
    
    n = list(set(t.graph.subjects(predicate=t.namespace['rank'],
                                object=t.namespace['rank/species'])))

    tr = Traits(directory='../eol/', verbose=True)

    conv = ne.convert(n, strip=True)
    converted = [(tr.namespace[i],k) for k,i in conv.items() if i != 'no mapping']
    tr.replace(converted)
    
    for i,g in enumerate([t.graph,tr.graph]):
        tmp = set([URIRef(a) for a in set(df['species'])])
        graph = get_subgraph(tmp, g, backtracking=0)
        graph.serialize('./data/taxonomy_%s.ttl' % str(i),format='ttl')
    
if __name__ == '__main__':
    #load_endpoint_data()
    #fingerprints()
    #chemical_features()
    #load_species_groups()
    #load_chemical_groups()
    #load_chemical_graph()
    load_taxonomy_graph()
    




































