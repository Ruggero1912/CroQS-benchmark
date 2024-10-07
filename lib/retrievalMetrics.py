import pandas as pd
import numpy as np
from typing import List


def compute_relevance_scores_all_queries(result_set : pd.DataFrame, ground_truth_query : str):
    """
    - df must be a result_set dataframe, as the one returned by RetrievalSystem.perform_query()
    - ground_truth_query must be the query whose result set is considered the ground truth of relevant documents
    returns a dataframe with an additional column 'relevance' that is valued 1 if the docno is in the result set 
    of the docs retrieved by ground_truth_query, else 0 
    """
    relevant_docs = result_set[result_set['query'] == ground_truth_query]['docno'].tolist()
    ret_result_set = result_set.copy()
    ret_result_set['relevance'] = ret_result_set['docno'].apply(lambda x: 1 if x in relevant_docs else 0)
    return ret_result_set

def compute_relevance_scores_given_relevant_docs(result_set : pd.DataFrame, relevant_docs : list):
    """
    - df must be a result_set dataframe, as the one returned by RetrievalSystem.perform_query()
    - relevant_docs must be the list of docnos that have to be considered relevant
    - returns a dataframe with an additional column 'relevance' that is valued 1 if the docno is in the result set 
    of the docs retrieved by ground_truth_query, else 0 
    """
    ret_result_set = result_set.copy()
    ret_result_set['relevance'] = ret_result_set['docno'].apply(lambda x: 1 if x in relevant_docs else 0)
    return ret_result_set

def _get_relevance_scores(df : pd.DataFrame, query : str, relevant_docs : list):
    """
    - df must be a result_set dataframe, as the one returned by RetrievalSystem.perform_query()
    - query must be the query for which you want to compute the relevance scores
    """
    result_set = df[df['query'] == query].copy()
    result_set['relevance'] = result_set['docno'].apply(lambda x: 1 if x in relevant_docs else 0)
    return result_set

def _dcg(relevances, k):
    """
    Compute the Discounted Cumulative Gain (DCG)
    - with b = 2
    """
    relevances = np.array(relevances)[:k]
    if relevances.size:
        return np.sum((2**relevances - 1) / np.log2(np.arange(1, relevances.size + 1) + 1))
    return 0.

def __dcg(relevances, k, b = 2):
    """
    Compute the Discounted Cumulative Gain (DCG)
    - with b = 2
    """
    relevances = np.array(relevances)[:k]
    if relevances.size:
        res = 0.
        for i, r in enumerate(relevances):
            res += ( r / max(1, np.emath.logn(b, i+1) ) )  # (2**r - 1)
        return res
    return 0.

def ndcg(result_set, k):
    """Compute the Normalized Discounted Cumulative Gain (NDCG)"""
    relevances = result_set['relevance'].tolist()
    ideal_relevances = sorted(relevances, reverse=True)
    div = _dcg(ideal_relevances, k)
    if div == 0.:
        print(f"[ndcg] [W] division by 0 [W]")
        return 0.
    return _dcg(relevances, k) / div

def average_precision(result_set : pd.DataFrame):
    """Compute the Average Precision (AP)"""
    relevances = result_set['relevance'].tolist()
    relevances = np.array(relevances)
    precisions = [np.sum(relevances[:i+1]) / (i+1) for i in range(len(relevances)) if relevances[i] > 0]
    if precisions:
        return np.mean(precisions)
    return 0.

#def map_score(result_set):
#    """
#    Compute the Mean Average Precision (MAP)
#    """
#    relevances = result_set['relevance'].tolist()
#    return average_precision(relevances)



def recall(result_set: pd.DataFrame, relevant_docs: list) -> float:
    """
    Compute the Recall of the result set.
    - result_set must be a dataframe containing the result set of a query.
    - relevant_docs must be the list of docnos that are considered relevant.
    - returns the recall score.
    """
    # Compute the number of relevant documents retrieved
    retrieved_relevant_docs = result_set['docno'].isin(relevant_docs).sum()
    
    # Compute the total number of relevant documents
    total_relevant_docs = len(relevant_docs)
    
    # Handle the case where there are no relevant documents to avoid division by zero
    if total_relevant_docs == 0:
        return 0.0
    
    # Compute recall
    recall_score = retrieved_relevant_docs / total_relevant_docs
    return recall_score

