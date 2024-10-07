from bert_score import score
from bert_score.scorer import BERTScorer


device="cuda:0"

BERT_SCORER_DISABLED = False

if not BERT_SCORER_DISABLED:
    bert_scorer = BERTScorer(lang="en", device=device)
else:
    bert_scorer = None
    print(f"[WARNING] BertScorer is disabled [WARNING]")

def compute_bert_score_p_r_f1(candidate_phrase : str, reference_phrase : str):
    if candidate_phrase is None or reference_phrase is None:
        return {
            "precision" : 0,
            "recall"    : 0,
            "f1-score"  : 0
        }
    # Define the candidate and reference texts
    candidates = [ candidate_phrase ]
    references = [ reference_phrase ]

    if not BERT_SCORER_DISABLED:
        # Compute BERTScore
        P, R, F1 = bert_scorer.score(candidates, references)    #, lang="en", verbose=False
    else:
        raise Exception("BertScorer is disabled")
    
    return {
        "precision" : round(P.mean().item(), 2),
        "recall"    : round(R.mean().item(), 2),
        "f1-score"  : round(F1.mean().item(), 2)
    }
    # Print the results
    #print(f"Precision: {P.mean().item():.4f}")
    #print(f"Recall: {R.mean().item():.4f}")
    #print(f"F1 Score: {F1.mean().item():.4f}")

import string

def remove_punctuation(sentence : str):
    # Create a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    # Use the translation table to remove punctuation from the sentence
    return sentence.translate(translator)

import nltk
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(sentence : str):
    global stop_words
    # Tokenize the sentence and filter out the stopwords
    filtered_words = [word for word in sentence.split() if word.lower() not in stop_words]
    return ' '.join(filtered_words)

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def stem_words(sentence : str):
    global stemmer
    # Tokenize the sentence and apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in sentence.split()]
    return ' '.join(stemmed_words)

def jaccard_similarity(sentence1 : str, sentence2 : str, remove_stopwords_toggle : bool = True, stem_words_toggle : bool = True):
    if sentence1 is None or sentence2 is None:
        return 0
    sentence1 = remove_punctuation(sentence1.lower())
    sentence2 = remove_punctuation(sentence2.lower())

    if remove_stopwords_toggle:
        sentence1 = remove_stopwords(sentence1)
        sentence2 = remove_stopwords(sentence2)

    if stem_words_toggle:
        sentence1 = stem_words(sentence1)
        sentence2 = stem_words(sentence2)
    # Tokenize the sentences into sets of words
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    
    # Compute the intersection and union of the sets
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    # Compute Jaccard similarity
    jaccard_sim = intersection / union if union != 0 else 0
    
    return round(jaccard_sim, 3)

from lib.retrievalSystem import RetrievalSystem
import numpy as np

def clip_similarity_sentence_coco_images(sentence : str, coco_ids : list, ir_system : RetrievalSystem) -> float:
    """
    
    """
    if sentence is None:
        print(f"[clip_similarity] received {sentence = }")
        return 0.0
    images_embeddings = ir_system.get_embeddings_from_dataset_ids(coco_ids, normalized=True)
    textual_embedding = ir_system.compute_text_embedding(sentence)
    #print("textual_query_embedding.shape:", textual_query_embedding.shape)
    #print("self.embeddings_dataset.shape:", self.embeddings_dataset.shape)
    result = np.mean( np.dot(images_embeddings, textual_embedding) )

    return round( result.item(), 3 )

def clip_similarity_sentences(sentence1 : str, sentence2 : str, ir_system : RetrievalSystem) -> float:
    """
    
    """
    if sentence1 is None or sentence2 is None:
        print(f"[clip_similarity] received {sentence1 = } {sentence2 = }")
        return 0.0
    textual_embedding1 = ir_system.compute_text_embedding(sentence1)
    textual_embedding2 = ir_system.compute_text_embedding(sentence2)
    #print("textual_query_embedding.shape:", textual_query_embedding.shape)
    #print("self.embeddings_dataset.shape:", self.embeddings_dataset.shape)
    result = np.dot(textual_embedding1, textual_embedding2)

    return round( result.item(), 3 )