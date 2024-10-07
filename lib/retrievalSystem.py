import h5py
import numpy as np
import os
from transformers import AutoTokenizer, CLIPModel
import pandas as pd
#from collections.abc import Callable

# here we can specify the dataset name that we want to use as collection for the retrieval system
H5PY_DATASET_NAME = "coco_val2017"
# here we must specify the path where the images associated to that dataset are stored
IMAGES_DATASET_PATH = "_data/dataset/coco/val2017"

# Normalize the embeddings
def normalize(v):
    was_1_dim = False
    if v.ndim == 1:
        was_1_dim = True
        v = v.reshape(1, -1)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    if was_1_dim:
        return ( v / norm ).squeeze()
    else:
        return v / norm

class RetrievalSystem:

    def __init__(self, embeddings_dataset, images_paths=None, ids_dataset=None, normalization : bool = True) -> None:
        #, callback : Callable[[int, dict], dict] = None
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.normalization = normalization
        self.embeddings_dataset = embeddings_dataset
        if self.normalization:
            self.normalized_embeddings_dataset = normalize(embeddings_dataset)
        self.images_paths = images_paths
        self.ids_dataset = ids_dataset
        #self.callback = callback


    def query(self, query, num_results=10, compute_mean_variance : bool = False) -> dict:
        """
        - param: query: str the textual query to perform
        - return: a dict whose keys are:
        - "indexes"
        - "scores"
        - "paths" if self.images_paths was given
        - "images_ids" if self.ids_dataset was given
        
        - the value associated to each key is a list whose length is equal at most to num_results
        """
        inputs = self.tokenizer([ query ], padding=True, return_tensors="pt", max_length=77, truncation=True) # limit the number of tokens to 77
        outputs = self.model.get_text_features(**inputs)
        #last_hidden_state = outputs.last_hidden_state
        #pooled_output = outputs.pooler_output  # pooled (EOS token) states
        textual_query_embedding = outputs[0].detach().numpy()

        if self.normalization:
            textual_query_embedding = normalize(textual_query_embedding)
            result = np.dot(self.normalized_embeddings_dataset, textual_query_embedding.T)
        else:
            result = np.dot(self.embeddings_dataset, textual_query_embedding.T)
        
        #print("textual_query_embedding.shape:", textual_query_embedding.shape)
        #print("self.embeddings_dataset.shape:", self.embeddings_dataset.shape)
        #result = np.dot(self.embeddings_dataset, textual_query_embedding.T)
        #assert self.embeddings_dataset.shape[1] == textual_query_embedding.shape[0], f"{self.embeddings_dataset.shape = } | {textual_query_embedding.shape = }"
        #print(f"{self.embeddings_dataset.shape = } | {textual_query_embedding.shape = } | {result.shape = }")
        #norm_sample = np.linalg.norm(self.embeddings_dataset[0:1], axis=1, keepdims=True)
        #norm_q = np.linalg.norm(textual_query_embedding.reshape(1,-1), axis=1, keepdims=True)
        #print(f"{norm_sample = } | {norm_q = }")
        #print(result)
        #print("result.shape: ", result.shape)

        sorted_results_idxs = np.argsort(result)[::-1][:num_results]

        #print("sorted indexes:", sorted_results_idxs)
        #print("sorted scalar products:", result[sorted_results_idxs])
        paths = []
        images_ids = []

        for pos, sorted_results_id in enumerate(sorted_results_idxs):

            if self.images_paths is not None:
                paths.append( self.images_paths[sorted_results_id])

            if self.ids_dataset is not None:
                images_ids.append(self.ids_dataset[sorted_results_id])

        #print( "paths len: ", len(paths), "\tsorted_results_idxs: ", len(sorted_results_idxs) )
        ret = {
            "indexes": sorted_results_idxs.tolist(),
            "scores" : result[sorted_results_idxs].tolist(),
            }
        
        if self.images_paths is not None and paths != None and len(paths) > 0:
            ret["paths"] = paths

        if self.ids_dataset is not None and images_ids != None and len(images_ids) > 0:
            ret["images_ids"] = images_ids

        if compute_mean_variance is True:
            mean_var = np.mean( np.var(self.embeddings_dataset[sorted_results_idxs], axis=0) ).item()
            ret["mean-var"] = mean_var
        
        return ret
    
    def _query_on_custom_dataset(self, query, custom_dataset_embeddings, custom_dataset_ids, num_results=10, compute_mean_variance : bool = False) -> dict:
        """
        - param: query: str the textual query to perform
        - return: a dict whose keys are:
        - "indexes"
        - "scores"
        - "paths" if self.images_paths was given
        - "images_ids" if self.ids_dataset was given
        
        - the value associated to each key is a list whose length is equal at most to num_results
        """
        inputs = self.tokenizer([ query ], padding=True, return_tensors="pt", max_length=77, truncation=True) # limit the number of tokens to 77
        outputs = self.model.get_text_features(**inputs)
        #last_hidden_state = outputs.last_hidden_state
        #pooled_output = outputs.pooler_output  # pooled (EOS token) states
        textual_query_embedding = outputs[0].detach().numpy()

        if self.normalization:
            textual_query_embedding = normalize(textual_query_embedding)
            custom_dataset_embeddings = normalize(custom_dataset_embeddings)
        
        result = np.dot(custom_dataset_embeddings, textual_query_embedding.T)
        
        sorted_results_idxs = np.argsort(result)[::-1][:num_results]

        images_ids = []

        for sorted_results_id in sorted_results_idxs:

            if self.ids_dataset is not None:
                images_ids.append(custom_dataset_ids[sorted_results_id])

        ret = {
            "indexes": sorted_results_idxs.tolist(),
            "scores" : result[sorted_results_idxs].tolist(),
            }

        if self.ids_dataset is not None and images_ids != None and len(images_ids) > 0:
            ret["images_ids"] = images_ids

        if compute_mean_variance is True:
            mean_var = np.mean( np.var(self.embeddings_dataset[sorted_results_idxs], axis=0) ).item()
            ret["mean-var"] = mean_var
        return ret
    
    def perform_query(self, queries, num_results : int = 100, compute_mean_variance : bool = False, custom_dataset_embeddings = None, custom_dataset_ids = None) -> pd.DataFrame:
        """
        - queries : str|list the query or list of queries to be executed
        - num_results : int the number of result documents to be returned
        - compute_mean_variance : bool if True, returns also the mean variance for each query in a column 'mean-var'
        - returns a dataframe whose keys are qid, query, docno, score, rank
        """
        
        if type(queries) == str:
            queries = [queries]

        results_df = pd.DataFrame(columns=['qid', 'query', 'docno', 'score', 'rank'] + (['mean-var'] if compute_mean_variance else []))

        df_offset = 0

        for qid, q in enumerate(queries):

            if custom_dataset_embeddings is None:
                results = self.query(q, num_results=num_results, compute_mean_variance=compute_mean_variance)
            else:
                results = self._query_on_custom_dataset(q, custom_dataset_embeddings, custom_dataset_ids, num_results=num_results, compute_mean_variance=compute_mean_variance)
            try:
                for i, result in enumerate( zip(results["images_ids"], results["scores"]) ):
                    #print(i+1, " ", result[0], " score: ", result[1])#, "path: ", img_path)
                    results_df.loc[df_offset] = [f"{qid}", q, f"{result[0]}", result[1], i] + ([results["mean-var"]] if compute_mean_variance else [])
                    #before df_offset there was  i  # other solution is qid * num_results + i 
                    df_offset += 1
            except Exception as e:
                print("Exception occurred !")
                print(e)
                print("results content: ", results_df)
                raise e
        
        return results_df
    
    def get_embeddings_from_dataset_indexes(self, indexes) -> np.ndarray:
        return  get_embeddings_from_dataset_indexes(self.embeddings_dataset, indexes)
    
    def get_embeddings_from_dataset_ids(self, ids, normalized : bool = False) -> np.ndarray:
        """
        returns a numpy array whose shape is (len(ids), embeddings_dataset.shape[1])
        - it means that every record of the returned array is an image embedding
        """
        embeddings = None

        for id in ids:  #here we must use the coco ids
            index = np.where(self.ids_dataset == id)[0][0]
            if normalized:
                if self.normalization:
                    tmp = self.normalized_embeddings_dataset[index]
                else:
                    tmp = normalize(self.embeddings_dataset[index])
            else:
                tmp = self.embeddings_dataset[index]
            if embeddings is None:
                embeddings = np.array([ tmp ])
            else:
                embeddings = np.concatenate((embeddings, [ tmp ] ) )
        
        return embeddings
    
    def compute_text_embedding(self, input_text : str) -> np.ndarray:
        assert isinstance(input_text, str)
        input_text = [input_text]
        inputs = self.tokenizer( input_text , padding=True, return_tensors="pt", max_length=77, truncation=True)  # limit the number of tokens to 77
        outputs = self.model.get_text_features(**inputs)
        textual_embedding = outputs[0]
        ret = textual_embedding.detach().numpy()
        if self.normalization:
            ret = normalize(ret)
        return ret


def get_datasets(HDF5_INDEX_FILE_PATH : str, H5PY_EMBEDDINGS_DATASET_NAME : str, H5PY_IDS_DATASET_NAME : str) -> tuple:
    """
    - returns (embeddings_dataset, dataset_ids) or None if something was not found
    """
    
    with h5py.File(HDF5_INDEX_FILE_PATH, 'a') as hf:
        if H5PY_EMBEDDINGS_DATASET_NAME in hf:
            embeddings_dataset = hf[H5PY_EMBEDDINGS_DATASET_NAME][:]
        else:
            print(f"dataset '{H5PY_EMBEDDINGS_DATASET_NAME}' not found ! ")
            return None

        if H5PY_IDS_DATASET_NAME in hf:
            dataset_ids = hf[H5PY_IDS_DATASET_NAME][:]
        else:
            print(f"dataset '{H5PY_IDS_DATASET_NAME}' not found ! ")
            return None
        
    return (embeddings_dataset, dataset_ids)

def get_embeddings_from_dataset_indexes(embeddings_dataset, indexes) -> np.ndarray:
    """
    - param the embeddings dataset 
    - param indexes: the list of indexes in the dataset of the embeddings to be returned 
    - returns a numpy array whose shape is (len(indexes), embeddings_dataset.shape[1])
    - it means that every record of the returned array is an image embedding
    """
    embeddings = None

    for idx in indexes:  #here we must use the indexes of the hfd5 dataset file
        if embeddings is None:
            embeddings = np.array([ embeddings_dataset[idx] ])
        else:
            embeddings = np.concatenate((embeddings, [embeddings_dataset[idx]] ) )
    
    return embeddings

def get_ir_system(dataType : str, HDF5_INDEX_FILE_PATH : str) -> RetrievalSystem:
    H5PY_EMBEDDINGS_DATASET_NAME = "coco_{}_embeddings".format(dataType)
    H5PY_IDS_DATASET_NAME = "coco_{}_ids".format(dataType)


    embeddings_dataset, dataset_ids = get_datasets(HDF5_INDEX_FILE_PATH, H5PY_EMBEDDINGS_DATASET_NAME, H5PY_IDS_DATASET_NAME)

    ir_system = RetrievalSystem(embeddings_dataset, ids_dataset=dataset_ids)
    return ir_system

if __name__ == "__main__":

    dataset = None

    print("Loading dataset")

    hf = h5py.File('_data/index/image_embeddings.h5', 'r')
    if H5PY_DATASET_NAME in hf:
        dataset = hf[H5PY_DATASET_NAME]
    else:
        print("Cannot load requested dataset")
        exit()


    dirpath = os.path.abspath(IMAGES_DATASET_PATH)
    image_paths = [ os.path.join(dirpath, foo) for foo in os.listdir(dirpath) ]

    print("Dataset loaded")


    print("Loading retrieval system")

    ir = RetrievalSystem(dataset, image_paths)

    print("Retrieval System loaded")

    input_txt = None

    while 1:

        input_txt = input("Your query:")
        if input_txt in ["", "!exit", "!q"]:
            break
        results = ir.query(input_txt, num_results=12)
        try:
            for i, result in enumerate( zip(results["paths"], results["scores"]) ):
                print(i+1, " ", result[0], " score: ", result[1])
        except Exception as e:
            print(e)
            
            print("results: ", results)
    
    hf.close()



