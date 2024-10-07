from typing import List

#from decap.decap import get_decap_model
#from decap.decapQueryExpansion import DeCapQueryExpansion
#from im2txtprojection.im2txtprojection import Im2TxtProjector, ProjectionType
from lib.retrievalSystem import RetrievalSystem
import numpy as np
#methods_to_load = [
#    {
#        "name" : "decap-centroid-msmarco",
#        "type" : "decap",
#        "apply-on" : "centroid", #"representative", "whole-cluster"
#        "custom-settings" : {"memory" : "MSMARCO"}
#    },
#    {
#        "name" : "clipcap-embeddingsSet-prompt1",
#        "type" : "clipcap",
#        "apply-on" : "whole-cluster", #"representative", "centroid"
#        "custom-settings" : {"prompt" : "Rewrite this query: ", "num-embeddings" : 4}
#    },
#    {
#        "name" : "groupcap-decap-COCO-prompt1",
#        "type" : "groupcap",
#        "apply-on" : "whole-cluster", #"representative", "centroid"
#        "custom-settings" : {"prompt" : "PROMPT-NAME", "num-captions" : 2, "captioning-method" : "decap"}
#    }
#]

class QSHandler:

    decap = None
    decapQE = None
    im2txt = None

    def __init__(self, ir_system : RetrievalSystem, device_str : str) -> None:  #methods_to_load : List[str], 
        self.ir_system = ir_system
        self.device = device_str
        pass
        #for method_infos in methods_to_load:
        #    assert isinstance(method_infos, dict)
        #    method_type = method_infos['type']
        #    if method_type == 'decap':
        #        projection_type = method_infos['custom-settings']['memory']
        #        self.load_decap(device_str, projection_type)
        #pass

    decap_qe_dict = {}

    #def load_decap(self, device : str, projection_type : str):
    #    """
    #    DEPRECATED
    #    """
    #    if projection_type not in self.decap_qe_dict:
    #        self.decap_qe_dict[projection_type] = DeCapQueryExpansion.load_object(device, projection_type)
    #    return self.decap_qe_dict[projection_type]
    
    def load_methods_from_qe_dict(self, qe_methods : dict):
        """
        WARNING - unimplemented method
        """
        print(F"[!] WARNING - unimplemented method [!]")
        
        required_keys = ['baseline', 'image-embeddings-centroids', 
                         'img-embeddings-representatives', 'img-embeddings-whole-cluster']
        
        for k in required_keys:
            if k not in qe_methods.keys():
                print(f"Missing required key '{k}'!")
                return None
        
        self.qe_methods = qe_methods

    def compute_suggested_query(self, original_query : str, coco_ids : list, method_dict : dict, method_type : str) -> str:
        """
        returns the string of the generated suggested query for the given set of images starting from the specified original query 
        using the specified method_name
        - method_type must be in ['img-embeddings-representatives', 'image-embeddings-centroids', 'img-embeddings-whole-cluster']
        - method_dict must contain the keys: ['name','object'] and also  'wants-query' if needed
        """
        
        most_representative_metric = 'cosine'

        cluster_embeddings = self.ir_system.get_embeddings_from_dataset_ids(coco_ids)
        # here i need to compute the centroid for the given coco_ids
        centroid = np.mean(cluster_embeddings, axis=0)
        if method_type == 'img-embeddings-representatives':

            assert most_representative_metric == 'cosine'

            def normalize(vectors):
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                return vectors / norms

            normalized_embeddings = normalize(cluster_embeddings)
            normalized_centroid = normalize(centroid.reshape(1, -1))
            cosine_similarities = np.dot(normalized_embeddings, normalized_centroid.T).flatten()
            most_representative_index = np.argmax(cosine_similarities)
            most_representative_embedding = cluster_embeddings[most_representative_index]
            most_representative_embedding = most_representative_embedding.reshape((1, -1))
            generated_q = run_query_suggestion_method(method_dict, most_representative_embedding, original_query, self.device)
            generated_q = generated_q[0] if isinstance(generated_q, list) else generated_q
            return generated_q
        elif method_type == 'image-embeddings-centroids':
            centroid = centroid.reshape((1,-1))      
            generated_q = run_query_suggestion_method(method_dict, centroid, original_query, self.device)
            generated_q = generated_q[0] if isinstance(generated_q, list) else generated_q
            return generated_q
        elif method_type == 'img-embeddings-whole-cluster':
            generated_q = run_query_suggestion_method(method_dict, cluster_embeddings, original_query)
            generated_q = generated_q[0] if isinstance(generated_q, list) else generated_q
            return generated_q
        elif method_type == 'baseline':
            print(f"[!] WARNING --- method_type == 'baseline' [!]")
            return "DISABLED-BASELINE"

import torch

import hashlib

def run_query_suggestion_method(method_dict : dict, data_parameter, original_query, device=None):

    #if 'clipcap' in method_dict['name']:
    #    digest = hashlib.sha256( data_parameter.data ).hexdigest()
    #    print(f"{method_dict = } \t\t {digest = }")
    
    if device is not None:
        data_parameter = torch.from_numpy( data_parameter ).to(device)
    
    #method_name = method_dict['name']
    #print(f"Computing QE for '{method_name}' - using {data_parameter.shape = } \t\t {original_query = }")
    #print(method_dict)

    if 'wants-query' in method_dict.keys():
        generated_query = method_dict['object'](data_parameter, original_query)
    else:
        generated_query = method_dict['object'](data_parameter)
    return generated_query





