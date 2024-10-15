import json

"""

json_data = [
        {
            "query" : "XXXX",
            "clusters" : {
                    "cluster-1" : {
                        "items" : [cocoid, cocoid, ...],
                        "caption" : "Human annotated suggested query"
                    },
                    "cluster-2" : {
                        ...
                    },
                    ...
        },
        ...
    ]

"""

from IPython.display import display, HTML
from jinja2 import Environment, FileSystemLoader

import pandas as pd

from lib.retrievalSystem import *
from lib.QSHandler import QSHandler

import pickle
import os

import requests

def save_coco_image(image_id, save_folder="images"):
    # Define the filename and ensure the ID is zero-padded to 12 characters
    filename = f"{str(image_id).zfill(12)}.jpg"
    save_path = os.path.join(save_folder, filename)

    # Check if the image already exists
    if os.path.exists(save_path):
        #print(f"Image {filename} already exists at {save_path}")
        return

    # Define the URL for the image
    url = f"http://images.cocodataset.org/train2017/{filename}"

    # Fetch the image content from the COCO URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Create the directory if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        
        # Save the image to the specified path
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        #print(f"Image saved to {save_path}")
    else:
        print(f"Failed to retrieve image. Status code: {response.status_code}")


jinja_env = Environment(loader=FileSystemLoader('templates'))
clusters_template = jinja_env.get_template('clusters.html')

class CroQS:

    queries = None

    def __init__(self, path) -> None:
        with open(path, 'r') as file:
            data = json.load(file)
            assert isinstance(data, list)

            self.queries = {}
            self.queries.clear()


            for query_infos_dict in data:
                assert isinstance(query_infos_dict, dict)

                assert "query" in query_infos_dict.keys()
                assert "clusters" in query_infos_dict.keys()

                query = query_infos_dict["query"]
                clusters_dict = query_infos_dict["clusters"]

                assert isinstance(clusters_dict, dict)

                self.queries[query] = clusters_dict

    def list_queries(self) -> list:
        return list(self.queries.keys())
    
    def list_clusters_queries(self, query) -> list:
        """
        returns a list of tuples, where each tuple is ("cluster-name", "human-defined-suggested-query")
        """
        return [ (cl_dict[0], cl_dict[1]["caption"]) for cl_dict in self.queries[query].items() ]
    
    def list_cluster_coco_ids(self, query, cluster):
        """
        returns the list of coco-ids associated with the specified cluster of the given initial query
        """
        return list( self.queries[query][cluster]["items"] )
    
    def show_clusters(self, query, render=False, images_local_path = None, images_path_from_html_folder = None, cors_proxy=None):
        """
        if render is true, render the html in a ipynb, else returns the html as a string
        """
        if images_local_path is not None:
            if not os.path.exists(images_local_path):
                print(f"the path '{images_local_path}' does not exist, ignoring")
                images_local_path = None
                images_path_from_html_folder = None
            else:
                for cluster in self._list_clusters_labels(query):
                    for coco_id in self.list_cluster_coco_ids(query, cluster):
                        save_coco_image(coco_id, images_local_path)
                if images_path_from_html_folder is None:
                    images_path_from_html_folder = images_local_path
                cors_proxy = None
            
        html = clusters_template.render(initial_query=query, clusters_dict=self.queries[query], 
                                        images_path_from_html_folder=images_path_from_html_folder,
                                        cors_proxy=cors_proxy)

        if render:
            display(HTML(html))
        else:
            return html

##################################################################

    evaluation_loaded = False

    suggested_queries = None

    device = None

    def evaluation_init(self, hdf5_index_file_path : str, device : str = "cuda:0", evaluation_dump_path : str = None):
        """
        call this method before using evaluation methods
        """
        self.evaluation_dump_path = evaluation_dump_path
        if CroQS.evaluation_loaded is True:
            print(f"[-] evaluation already loaded [-]")
            return
        
        CroQS.ir_system = get_ir_system("train2017", hdf5_index_file_path)
        self._load()
        if CroQS.suggested_queries is None:
            print(f"No dict object loaded from disk, creating new empty dict")
            CroQS.suggested_queries = {}
        
        CroQS.device = device
        
        CroQS.evaluation_loaded = True

    def _list_clusters_labels(self, query):
        return list( self.queries[query].keys() )
    
    def _get_coco_ids_of_query(self, query):
        """
        returns the list of coco-ids of the original result set of the specified query
        """
        coco_ids = []
        for cluster_label in self._list_clusters_labels(query):
            coco_ids += self.list_cluster_coco_ids(query, cluster_label)
        return coco_ids
    
    def _get_human_annotation_for(self, query, cluster) -> str:
        """
        returns the human-defined annotation defined in the benchmark for the specified query - cluster 
        """
        return self.queries[query][cluster]["caption"]

    qs_common_interface = None

    def _get_qs_common_interface(self):
        if self.qs_common_interface is None:
            self.qs_common_interface = QSHandler(self.ir_system, self.device)
        return self.qs_common_interface

    def _save(self):
        """
        should store on disk the self.suggested_queries dict object
        """
        if self.evaluation_dump_path is None:
            return
        with open(self.evaluation_dump_path, 'wb') as f:
            pickle.dump(self.suggested_queries, f)

    def _load(self):
        """
        should load from disk the previously stored dict object, if any
        """
        if self.evaluation_dump_path is None:
            print(f"no dump path provided")
            return
        if os.path.exists(self.evaluation_dump_path):
            with open(self.evaluation_dump_path, 'rb') as f:
                CroQS.suggested_queries = pickle.load(f)
        else:
            print(f"File {self.evaluation_dump_path} not found")

    def load_query_suggestions(self, query, cluster_label, qs_dict : dict, force_update : bool = False, compute_scores : bool = True,
                              do_save : bool = True) -> dict:
        """
        computes the query suggestions for the specified query and clusters with all the methods specified in qs_dict
         - qs_dict must be built according to the structure in methods.py
         - if a suggested-query for a method already exists, it is not recomputed
         - returns a dict ('method-name' -> 'suggested-query') with all the computed suggested-queries (not necessarily in qs_dict)
        """

        if query not in self.list_queries():
            print(F"The specified query is not in the list {query = }")
            return None
        
        if cluster_label not in self._list_clusters_labels(query):
            print(F"The specified {cluster_label = } is not in the list for the query {query = }")
            return None
        
        #clustering_type = self.get_clustering_type(query)

        qs_common_interface = self._get_qs_common_interface()

        coco_ids = self.list_cluster_coco_ids(query, cluster_label)

        if query not in self.suggested_queries.keys():
            self.suggested_queries[query] = {}

        if cluster_label not in self.suggested_queries[query].keys():
            self.suggested_queries[query][cluster_label] = {}

        if "suggested-queries" not in self.suggested_queries[query][cluster_label].keys():
            self.suggested_queries[query][cluster_label]["suggested-queries"] = {}

        changed = False

        score_computation_for = []
        
        for method_type, methods_list in qs_dict.items():
            #if method_type == 'baseline': print("WARNING baseline method without imports"); continue
            for method_dict in methods_list:
                method_name = method_dict['name']
                if force_update or method_name not in self.suggested_queries[query][cluster_label]["suggested-queries"].keys() \
                    or self.suggested_queries[query][cluster_label]["suggested-queries"][method_name]['query'] is None:
                    sugg_query = qs_common_interface.compute_suggested_query(query, coco_ids, method_dict, method_type)
                    self.suggested_queries[query][cluster_label]["suggested-queries"][method_name] = {"query" : sugg_query}
                    changed = True
                    score_computation_for.append(method_name)
                else:
                    print(f"\rLoading already existing qs for the method '{method_name}'", end="")
                    sugg_query = self.suggested_queries[query][cluster_label]["suggested-queries"][method_name]["query"]
                    if compute_scores is True:
                        if 'scores' not in self.suggested_queries[query][cluster_label]["suggested-queries"][method_name].keys() or self.suggested_queries[query][cluster_label]["suggested-queries"][method_name]["scores"] is None:
                            print(F"the scores for the method '{method_name}' were not computed, setting changed to true... ")
                            changed = True
                            score_computation_for.append(method_name)

        if changed is True: 
            if compute_scores:
                #print(f"Computing scores for cluster '{cluster_label}' of {query = }")
                self.compute_scores_query_suggestions(query, cluster_label, do_save=False, score_computation_for=score_computation_for)
            if do_save:
                self._save()

        ret_dict = {}

        for method_name, infos in self.suggested_queries[query][cluster_label]["suggested-queries"].items():
            ret_dict[method_name] = infos["query"]

        return ret_dict, changed

    qs_common_interface = None

    def get_query_suggestions(self, query, cluster_label) -> dict:
        """
        returns the dictionary of the currently computed query suggestions (key: method_name, value: suggested_query string)
        """

        if query not in self.list_queries():
            print(F"The specified query is not in the list {query = }")
            return None
        
        if cluster_label not in self._list_clusters_labels(query):
            print(F"The specified {cluster_label = } is not in the list for the query {query = }")
            return None

        ret_dict = {}

        for method_name, infos in self.suggested_queries[query][cluster_label]["suggested-queries"].items():
            ret_dict[method_name] = infos["query"]

        return ret_dict
    
    def get_query_suggestions_dicts(self, query, cluster_label) -> dict:
        """
        returns the dictionary of the currently computed query suggestions (key: method_name, value: dict{'query' : exp_q_string, 'scores' : scores_dictionary})
        - at the method_name "self.DEFAULT_EXP_Q_KEY = '__gt-qs'" there is the ground truth human generated suggested query
        """

        if query not in self.list_queries():
            print(F"The specified query is not in the list {query = }")
            return None
        
        if cluster_label not in self._list_clusters_labels(query):
            print(F"The specified {cluster_label = } is not in the list for the query {query = }")
            return None

        if "suggested-queries" not in self.suggested_queries[query][cluster_label].keys():
            print(f"[!](get_query_suggestions_dicts) 'suggested-queries' were not generated still [!]")
            return None

        return self.suggested_queries[query][cluster_label]["suggested-queries"].copy()
    
    def get_query_suggestions_dataframe_for_query(self, query, qs_dict, allowed_methods_names : list = None, do_save : bool = True) -> pd.DataFrame:
        """
        returns the results dataframe for the specified query
        - returns Tuple(DataFrame, bool changed)
        """
        if query not in self.list_queries():
            print(F"The specified query is not in the list {query = }")
            return None
        
        results_df = None #pd.DataFrame(columns=['query', 'cluster', 'method', 'suggested-query'])

        no_changes = True
        
        for cluster_label in self._list_clusters_labels(query):
            data, changed = self.load_query_suggestions(query, cluster_label, qs_dict, do_save=True)
            if changed is True:
                no_changes = False
            cluster_query_suggestions_dict = self.get_query_suggestions_dicts(query, cluster_label)
            if self.DEFAULT_EXP_Q_KEY not in cluster_query_suggestions_dict.keys() or 'scores' not in cluster_query_suggestions_dict[self.DEFAULT_EXP_Q_KEY].keys():
                print(f"restoring qses for init-q '{query}' cluster '{cluster_label}'")
                data, changed = self.load_query_suggestions(query, cluster_label, qs_dict, force_update=True, do_save=True)
                if changed is True:
                    no_changes = False
                cluster_query_suggestions_dict = self.get_query_suggestions_dicts(query, cluster_label)
            
            if cluster_query_suggestions_dict[self.DEFAULT_EXP_Q_KEY] == 'disabled':
                print(f"Skipping {cluster_label = } for {query = } since this cluster is disabled")
                continue

            for method_name, method_infos in cluster_query_suggestions_dict.items():
                if allowed_methods_names is not None and method_name not in allowed_methods_names:
                    continue
                row_infos = {
                    'query'         : query,
                    'cluster'       : cluster_label,
                    'method'        : method_name,
                    'suggested-query': method_infos['query'],
                    **method_infos['scores']
                }
                if results_df is None:
                    results_df = pd.DataFrame( columns=row_infos.keys() )
                results_df.loc[len(results_df)] = row_infos#results_df.append(row_infos, ignore_index=True)
        
        if no_changes is False:
            if do_save:
                self._save()
        
        return results_df, (not no_changes)
    
    def get_query_suggestions_dataframe_all_queries(self, qs_dict, allowed_methods_names : list = None) -> pd.DataFrame:
        """
        returns the dataframe with the scores for all the queries, 
        - internally calls get_query_suggestions_dataframe_for_query
        """
        no_changes = True
        ret_df = None
        try:
            from tqdm import tqdm
        except:
            def tqdm(input):
                return input
    
        for q in tqdm(self.list_queries()):

            tmp, changed = self.get_query_suggestions_dataframe_for_query(q, qs_dict, allowed_methods_names, do_save=False)
            assert isinstance(tmp, pd.DataFrame)
            if changed is True:
                no_changes = False
            if ret_df is None:
                ret_df = tmp
            else:
                ret_df = pd.concat([ret_df, tmp], ignore_index=True)
        if not no_changes:
            self._save()
        return ret_df



    DEFAULT_EXP_Q_KEY = 'human'

    def compute_scores_query_suggestions(self, query : str, cluster_label : str, 
                                        num_results : int = 100, ndcg_k : int = 10,
                                        do_save : bool = True,
                                        score_computation_for : list = None) -> dict:
        """
        compute the scores for all the methods of the specified cluster and saves them
        - if score_computation_for is not None, computes the scores only for the method specified in score_computation_for list
        """
        from lib.retrievalMetrics import ndcg, recall, average_precision, compute_relevance_scores_all_queries, compute_relevance_scores_given_relevant_docs
        from lib.sentenceSimilarity import jaccard_similarity, clip_similarity_sentence_coco_images, clip_similarity_sentences
        
        if query not in self.list_queries():
            print(F"The specified query is not in the list {query = }")
            return None
        
        if cluster_label not in self._list_clusters_labels(query):
            print(F"The specified {cluster_label = } is not in the list for the query {query = }")
            return None

        default_suggested_query = self._get_human_annotation_for(query, cluster_label)

        if default_suggested_query is None:
            print(F"[!] The default suggested query is None for the cluster '{cluster_label}' of the query '{query}' [!]")

        query_suggestions_dict = self.get_query_suggestions(query, cluster_label)

        cluster_coco_ids = self.list_cluster_coco_ids(query, cluster_label)

        if score_computation_for is None:
            tmp_queries_list = set([default_suggested_query] + list(query_suggestions_dict.values()))
        else:
            tmp_queries_list = set([default_suggested_query])
            for method_name, exp_q in query_suggestions_dict.items():
                if method_name not in score_computation_for:
                    continue
                tmp_queries_list.add(exp_q)

        search_results = self.ir_system.perform_query( tmp_queries_list , num_results=num_results )

        original_query_coco_ids = self._get_coco_ids_of_query(query)
        original_query_embeddings = self.ir_system.get_embeddings_from_dataset_ids(original_query_coco_ids)
        search_results_closed_set = self.ir_system.perform_query(tmp_queries_list, num_results=len(original_query_coco_ids), 
                                                                 custom_dataset_embeddings=original_query_embeddings, 
                                                                 custom_dataset_ids=original_query_coco_ids)
        result_set_relevance_judgements_on_GT_query = compute_relevance_scores_all_queries(search_results, default_suggested_query)
        cluster_doc_ids = [ str(el) for el in cluster_coco_ids]
        result_set_relevance_judgements_on_GT_cluster = compute_relevance_scores_given_relevant_docs(search_results, cluster_doc_ids)
        
        self.suggested_queries[query][cluster_label]["suggested-queries"][self.DEFAULT_EXP_Q_KEY] = {'query' : default_suggested_query}

        generated_queries_similiarity_metrics = {}
        for method_name, infos in self.suggested_queries[query][cluster_label]["suggested-queries"].items():

            if method_name not in score_computation_for and method_name != self.DEFAULT_EXP_Q_KEY:
                continue

            suggested_query = infos['query']

            res_set_exp_q = result_set_relevance_judgements_on_GT_query[result_set_relevance_judgements_on_GT_query['query'] == suggested_query]
            res_set_exp_q_on_cluster = result_set_relevance_judgements_on_GT_cluster[result_set_relevance_judgements_on_GT_cluster['query'] == suggested_query]
            
            res_set_closed_set_exp_q = search_results_closed_set[search_results_closed_set['query'] == suggested_query ]

            recall_at = len(cluster_doc_ids)

            infos['scores'] = {
                #"bert-score-wrt-GT-suggested-query" : compute_bert_score_p_r_f1(suggested_query, cluster_default_caption),
                #"jaccard-expq"    : jaccard_similarity(suggested_query, default_suggested_query, True, True),
                "jaccard-q"    : jaccard_similarity(suggested_query, query, True, True),
                "clip-similarity-cluster" : clip_similarity_sentence_coco_images(suggested_query, cluster_coco_ids, self.ir_system),
                "clip-similarity-q" : clip_similarity_sentences(suggested_query, query, self.ir_system),
                #"clip-similarity-expq" : clip_similarity_sentences(suggested_query, default_suggested_query, self.ir_system),
                #f"ndcg@{ndcg_k}-expq" : ndcg(res_set_exp_q, k=ndcg_k),
                #"MAP-expq" : average_precision(res_set_exp_q),
                f"NDCG@{ndcg_k}" : ndcg(res_set_exp_q_on_cluster, k=min(ndcg_k, len(cluster_coco_ids))),
                "MAP" : average_precision(res_set_exp_q_on_cluster),
                f"Recall-Open-Set-@{num_results}" : recall(res_set_exp_q, cluster_doc_ids),
                "Recall-Closed-Set" : recall(res_set_closed_set_exp_q[0:recall_at], cluster_doc_ids),
                #f"recall@len(orig_res_set)-CLOSED-SET-cluster" : recall(res_set_closed_set_exp_q, cluster_doc_ids),
            }
        if do_save:
            self._save()

    default_scores_to_show = [
        #"method",
        "jaccard-q",
        "clip-similarity-cluster",
        "clip-similarity-q",
        #"clip-similarity-expq",
        "MAP",
        "Recall-Open-Set-@100",
        "Recall-Closed-Set"
    ]

