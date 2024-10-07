from lib.methods.clipcap.clipcap import *

"""
def get_generated_captions_query_expansion(a, q):
    return "UNIMPLEMENTED"

def get_generated_captions(a, q=None):
    return "UNIMPLEMENTED"
"""

import torch

from lib.methods.decap.DecapQueryExpansion import DeCapQueryExpansion


class Utils:
    device = None

from lib.sorting import dissimilarity_ranking


def build_expanded_query_from_most_representatives_within_cluster(cluster_embeddings, initial_query : str, TOP_K : int = 5):
    #print(type(cluster_embeddings))
    #print(cluster_embeddings.shape)
    most_repr = dissimilarity_ranking(cluster_embeddings)
    top_embs = torch.from_numpy( cluster_embeddings[most_repr[:TOP_K]] ).to(Utils.device)
    return get_generated_caption_query_expansion_on_top_cluster_items(top_embs, initial_query)

def build_expanded_query_from_most_representatives_within_cluster_without_query(cluster_embeddings, initial_query : str = ""):
    return build_expanded_query_from_most_representatives_within_cluster(cluster_embeddings, "")

def build_expanded_query_from_most_representatives_within_cluster_prompting_1(cluster_embeddings, initial_query : str):
    prompt = f"Rewrite this sentence as a query: {initial_query} "
    return build_expanded_query_from_most_representatives_within_cluster(cluster_embeddings, prompt)

def build_expanded_query_from_most_representatives_within_cluster_prompting_2(cluster_embeddings, initial_query : str):
    prompt = f"Rewrite this query: {initial_query} "
    return build_expanded_query_from_most_representatives_within_cluster(cluster_embeddings, prompt)

METHODS_GROUPS = [
    "baseline",
    "image-embeddings-centroids",
    "img-embeddings-representatives",
    "img-embeddings-whole-cluster"
]

import re

def get_qe_methods_dict(methods_keys : list, bo1 = None, colbert_query_encoder = None, colbert_prf = None, decap_qe : DeCapQueryExpansion = None, device=None):
    """
    - returns the dict of methods to use according to the form required by evaluators' methods
    - param bo1 : the pyterrier bo1 object to use for query expansion
    - param colbert_prf : the pyterrier-colbert prf object
    - param methods_keys : the list of methods to use:
        - you can use the keyword 'all' to return all the methods (e.g. methods_keys='all')
        - you can use the keyword 'default' to load only a set of default methods
        - you can use the name of one of the groups of methods to return all the methods belonging to that group (e.g. methods_keys=['baseline'])
    """
    if device is None:
        print(f"WARNING you did not specify the device WARNING")
    
    Utils.device = device

    def bo1_method(result_df) -> str:
        qe_tmp = bo1.transform(result_df)
        pattern = r'\b(\w+)\^\d+\.\d+\b'
        # Find all matches of the pattern
        matches = re.findall(pattern, qe_tmp.iloc[0]['query'])
        return " ".join( matches ) # qe_tmp.iloc[0]['query']#['new_query_stemmed']


    def bo1_method_prompting(result_df, initial_query : str) -> tuple:
        qe_tmp = bo1.transform(result_df)
        #qe_tmp = qe_tmp.iloc[0]['query']#['new_query_stemmed']
        #qe_tmp = " ".join( qe_tmp.iloc[0]['query'] )
        pattern = r'\b(\w+)\^\d+\.\d+\b'
        # Find all matches of the pattern
        matches = re.findall(pattern, qe_tmp.iloc[0]['query'])
        qe_tmp = " ".join( matches )
        # here i should iterate over the stemmed tokens given by bo1 and remove the ones that
        # are already present in the original query,
        init_q_words = []
        import pyterrier as pt
        po = pt.TerrierStemmer.porter
        for word in initial_query.split(" "):
            init_q_words.append( po.stem(word) )
        
        add_ctx = []
        for q_w in qe_tmp.split(" "):
            if q_w not in init_q_words:
                add_ctx.append(q_w)
        add_ctx = " ".join(add_ctx)
        # then i should build the new query like:
        return (initial_query, add_ctx)#f"{initial_query} with this additional context: {add_ctx}"

    def bo1_method_prompting_1(result_df, initial_query : str) -> str:
        (initial_query, add_ctx) = bo1_method_prompting(result_df, initial_query)
        return f"{initial_query} with this additional context: {add_ctx}"

    def bo1_method_prompting_2(result_df, initial_query : str) -> str:
        (initial_query, add_ctx) = bo1_method_prompting(result_df, initial_query)
        return f"{initial_query} in this context: {add_ctx}"

    def bo1_method_prompting_3(result_df, initial_query : str) -> str:
        (initial_query, add_ctx) = bo1_method_prompting(result_df, initial_query)
        return f"{initial_query} {add_ctx}"

    def bo1_method_prompting_4(result_df, initial_query : str) -> str:
        (initial_query, add_ctx) = bo1_method_prompting(result_df, initial_query)
        return f"{initial_query}. keywords: {add_ctx}"
    
    def colbert_prf_method(result_df):
        qe_tmp = colbert_prf.transform(colbert_query_encoder(result_df))
        return qe_tmp.iloc[0]['query'] + " " + ( " ".join( qe_tmp.iloc[0]['query_toks'] ) )




    qe_methods ={
            "baseline" : [
                {
                    "name"      : "bo1",
                    "object"    : bo1_method,
                    "type"      : "bo1"
                },
                {
                    "name"          : "bo1-prompting-1",
                    "object"        : bo1_method_prompting_1,
                    "wants-query"   : True,
                    "type"          : "bo1"
                },
                {
                    "name"          : "bo1-prompting-2",
                    "object"        : bo1_method_prompting_2,
                    "wants-query"   : True,
                    "type"          : "bo1"
                },
                {
                    "name"          : "bo1-prompting-3",
                    "object"        : bo1_method_prompting_3,
                    "wants-query"   : True,
                    "type"          : "bo1"
                },
                {
                    "name"          : "bo1-prompting-4",
                    "object"        : bo1_method_prompting_4,
                    "wants-query"   : True,
                    "type"          : "bo1"
                },
                {
                    "name"          : "colbert-prf",
                    "object"        : colbert_prf_method,
                    "type"          : "colbert-prf",
                    "default"       : True,
                }
            ],
            "image-embeddings-centroids" : [
                {
                    "name"      : "clipcap-on-centroids",
                    "object"    : get_generated_captions,
                    "default"   : True,
                },
                {
                    "name"          : "initial-query",
                    "object"        : (lambda img_emb, init_q : init_q),
                    "wants-query"   : True,
                    #"type"          : "bo1"
                },
                {
                    "name"      : "clipcap-prompting-on-centroids",
                    "object"    : get_generated_captions_query_expansion,
                    "wants-query"   : True
                },
                {
                    "name"      : "decap-on-centroids",
                    "object"    : decap_qe.get_generated_captions if decap_qe is not None else None,
                    "type"      : "decap",
                    "default"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.0",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.0))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.05",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.05))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.1",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.1))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.2",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.2))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.3",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.3))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.4",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.4))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.45",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.45))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.5",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.5))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.55",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.55))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.6",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.6))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.65",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.65))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.7",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.7))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.75",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.75))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.8",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.8))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-0.9",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=0.9))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                },
                {
                    "name"      : "decap-on-centroids-projection-1.0",
                    "object"    : (lambda image_embeddings, initial_queries: decap_qe.get_generated_query_expansions(image_embeddings, initial_queries, beta=1.0))  if decap_qe is not None else None,
                    "type"      : "decap",
                    "wants-query"   : True,
                }
            ],
            "img-embeddings-representatives" : [
                {
                    "name"      : "clipcap-on-representatives",
                    "object"    : get_generated_captions
                },
                {
                    "name"      : "clipcap-prompting-on-representatives",
                    "object"    : get_generated_captions_query_expansion,
                    "wants-query"   : True
                },
                {
                    "name"      : "decap-on-representatives",
                    "object"    : decap_qe.get_generated_captions if decap_qe is not None else None,
                    "type"      : "decap"
                }
            ],
            "img-embeddings-whole-cluster" : [
                {
                    'name' : 'clipcap-query_set-of-embeddings',
                    'object' : build_expanded_query_from_most_representatives_within_cluster,
                    'wants-query' : True,
                    'type'  : 'clipcap'
                },{
                    'name' : 'clipcap-prompt_query_embeddings-set__1',
                    'object' : build_expanded_query_from_most_representatives_within_cluster_prompting_1,
                    'wants-query' : True,
                    'type'  : 'clipcap'
                },{
                    'name' : 'clipcap-prompt_query_embeddings-set__2',
                    'object' : build_expanded_query_from_most_representatives_within_cluster_prompting_2,
                    'wants-query' : True,
                    'type'  : 'clipcap'
                },{
                    'name'      : 'clipcap-embeddings-set__no-query',
                    'object'    : build_expanded_query_from_most_representatives_within_cluster_without_query,
                    'wants-query' : True,
                    'type'      : 'clipcap',
                    'default'   : True,
                },
            ]
        }
        
    allowed_types = ['clipcap']

    if bo1 is not None:
        allowed_types.append('bo1')

    if colbert_prf is not None and colbert_query_encoder is not None:
        allowed_types.append('colbert-prf')

    if decap_qe is not None:
        allowed_types.append("decap")

    #if methods_keys == 'all':
    #    if bo1 is None:
    #        print("[x] ERROR bo1 is None but all methods were requested [x]")
    #        return None
    #    return qe_methods

    
    for methods_group_name in qe_methods.keys():
        #if methods_group_name in methods_keys:
            #print(f'methods group "{methods_group_name}" in methods_keys, skipping checks')
        #    continue
        to_remove = []
        for method in qe_methods[methods_group_name]:
            if methods_keys != 'all' and method['name'] not in methods_keys and methods_group_name not in methods_keys:
                #print(f"the method {method['name']} is not in methods_keys, removing")
                if methods_keys != 'default' or 'default' not in method.keys() or method['default'] != True:
                    to_remove.append(method)
                #qe_methods[methods_group_name].remove(method)
            elif 'type' in method.keys() and method['type'] not in allowed_types:
                to_remove.append(method)
                if methods_group_name in methods_keys:
                    print(f"removing method '{method['name']}' since required parameter was not given for type '{method['type']}'")
        for to_r in to_remove:
            qe_methods[methods_group_name].remove(to_r)
    
    return qe_methods


def add_method_to_dict(qe_dict : dict, method_name : str, method : callable, 
               wants_query : bool = False, type : str = None, 
               method_group_name : str = "image-embeddings-centroids"):
    
    if method_group_name not in qe_dict.keys():
        print(f"[!] {method_group_name=} is not in {qe_dict.keys()=} [!]")
        return qe_dict
    
    for el in qe_dict[method_group_name]:
        if el['name'] == method_name:
            print(f"already exists a method with name = {method_name}\t returning dict unchanged")
            return qe_dict
    
    method_dict = {
        "name"      : method_name,
        "object"    : method,   
    }

    if wants_query != False:
        method_dict['wants-query'] = True
    if type != None:
        method_dict['type'] = type
    
    qe_dict[method_group_name].append(method_dict)
    return qe_dict