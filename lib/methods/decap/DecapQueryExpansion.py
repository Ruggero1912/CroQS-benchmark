from lib.methods.decap.decap import DeCap, Decoding, get_decap_model
from lib.methods.decap.Im2TxtProjection import Im2TxtProjector, ProjectionType
from typing import List

class DeCapQueryExpansion:

    beta = 0.7

    def __init__(self, decap_model : DeCap, im2txt : Im2TxtProjector, beta : float = None) -> None:
        """
        - beta must be within 0 and 1, it is used in method transform() if text is not None and in method get_generated_query_expansions
        - beta == 1 --> projection only depends on input image
        - beta == 0 --> projection only depends on input text
        """
        self.decap_model = decap_model
        self.im2txt = im2txt

    def transform(self, image_features, text : str = None, beta : float = None):
            
        prefix_embedding = self.im2txt.project(image_features)

        generated_text = Decoding(self.decap_model,prefix_embedding)
        generated_text = generated_text.replace('<|startoftext|>','').replace('<|endoftext|>','')
        return generated_text
    
    def get_generated_query_expansions(self, image_embeddings, initial_queries, beta = None) -> List[str]:
        """
        Build an expanded query for each given embedding
        - image_embeddings must be a ndarray of shape (NUM_EMBEDDINGS, EMBEDDING_SIZE)
        - if initial_queries is a str, assert that the same initial_query has to be used for all the images
        - else len(image_embeddings) must be equal to len(initial_queries)
        - returns a List[str] containing the generated caption for each embedding
        - beta must be within 0 and 1, it is used in method transform()
        - beta == 1 --> projection only depends on input image
        - beta == 0 --> projection only depends on input text
        """
        if initial_queries is None or initial_queries == "":
            print(f"get_generated_query_expansions received {initial_queries = } going to call get_generated_captions")
            return self.get_generated_captions(image_embeddings)
        
        if type(initial_queries) == str:
            initial_queries = [initial_queries] * len(image_embeddings)

        results = []
        
        for image_embedding, initial_query in zip(image_embeddings, initial_queries):
            results.append( self.transform(image_embedding, initial_query, beta) )

        return results
    
    def get_generated_captions(self, image_embeddings) -> List[str]:
        """
        Builds a caption for each given image embedding
        - image_embeddings must be a ndarray of shape (NUM_EMBEDDINGS, EMBEDDING_SIZE)
        - returns a List[str] containing the generated caption for each embedding
        """
        results = []
        
        for image_embedding in image_embeddings:
            results.append( self.transform(image_embedding) )

        return results
    
    allowed_projection_types = [
        'coco',
        'msmarco'
    ]

    im2txt_dict = {

    }
    decap = None

    def load_object(device : str, projection_type : str, decap_object = None):
        """
        static method that returns an instance of decapQueryExpansion
        - allowed projection types: DeCapQueryExpansion.allowed_projection_types ['coco', 'msmarco']
        """
        projection_type_obj = None
        if projection_type == 'coco':
            projection_type_obj = ProjectionType.COCO_CAPTIONS
        elif projection_type == 'msmarco':
            projection_type_obj = ProjectionType.MS_MARCO_QUERIES_A
        else:
            print(f"[load_object] Using default projection type '{ProjectionType.COCO_CAPTIONS = }' since '{projection_type = }' was not recognized")
            print(f"[load_object] {DeCapQueryExpansion.allowed_projection_types = } ")
            projection_type = 'coco'
            projection_type_obj = ProjectionType.COCO_CAPTIONS
        if DeCapQueryExpansion.decap is None:
            if decap_object is not None:
                DeCapQueryExpansion.decap = decap_object
            else:
                DeCapQueryExpansion.decap = get_decap_model(device)
        # here we should assert that the decap obj device is the same as device-str
        #if DeCapQueryExpansion.im2txt is None:
        #    DeCapQueryExpansion.im2txt = Im2TxtProjector(projection_type_obj, device_str=device)
        if projection_type not in DeCapQueryExpansion.im2txt_dict.keys():
            im2txt = Im2TxtProjector(projection_type_obj, device_str=device)
            DeCapQueryExpansion.im2txt_dict[projection_type] = im2txt
        else:
            im2txt = DeCapQueryExpansion.im2txt_dict[projection_type]
        #if DeCapQueryExpansion.decapQE is None:
        #    DeCapQueryExpansion.decapQE = DeCapQueryExpansion(DeCapQueryExpansion.decap, DeCapQueryExpansion.im2txt)
        decapQE = DeCapQueryExpansion(DeCapQueryExpansion.decap, im2txt)
        return decapQE
