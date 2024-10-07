from enum import Enum
import numpy as np
import json
import random
import torch
from tqdm import tqdm
import os
import h5py
from typing import Tuple

class ProjectionType(Enum):
    COCO_CAPTIONS       = 'coco_captions'
    MS_MARCO_QUERIES_A  = 'ms_marco_queries_a'

class Im2TxtProjector:

    SUPPORT_MEMORY_SIZE = 50000
    EMBEDDINGS_DIMENSION = 512

    COCO_DATASET_DIR_PATH = os.path.join( os.getenv('DATA_DIRECTORY_ROOT'), 'coco-dataset')
    captions_dataType = 'train2017'
    ANNOTATIONS_CAPTION_FILE_PATH = os.path.join(COCO_DATASET_DIR_PATH, 'annotations/captions_{}.json'.format(captions_dataType) )

    MS_MARCO_QUERIES_DATASET_DIR_PATH = os.path.join( os.getenv('DATA_DIRECTORY_ROOT'), 'MSMarco/queries')
    MS_MARCO_QUERIES_FILE_PATH = os.path.join(MS_MARCO_QUERIES_DATASET_DIR_PATH, 'queries.train.tsv')

    def __init__(self, type : ProjectionType = ProjectionType.COCO_CAPTIONS, verbose : bool = True, device_str = "cpu", ir_system = None) -> None:
        """
        if ir_system is given, can compute projections that consider also an original text
        """
        # check if hdf5 already exists, otherwhise builds the support memory for that kind
        
        #if type not in ProjectionType.mro()

        self.type = type
        self.device_str = device_str
        self.device = torch.device(self.device_str)
        self.H5PY_FILE_PATH = os.path.join(os.getenv('REPO_DIRECTORY_ROOT'), 'code/_data/index/{}_text_embeddings.h5'.format(type.value) )
        self.H5PY_EMBEDDINGS_DATASET_NAME = '{}-embeddings'.format(type.value)
        self.H5PY_TEXT_DATASET_NAME = '{}-text'.format(type.value)

        embs_dataset, text_dataset = self._load_support_memory()

        if text_dataset is None:
            if verbose: print(f"[+] Going to build support memory for the given data type: {type} [+]")
            embs_dataset, text_dataset = self._build_support_memory()
            if verbose: print(f"[+] Done [+]")
        
        self.text_dataset = text_dataset
        self.embs_dataset = torch.tensor(embs_dataset[:]).to(self.device)
        if ir_system is not None:
            from lib.retrievalSystem import RetrievalSystem
            assert isinstance(ir_system, RetrievalSystem)
        self.ir_system = ir_system
        


    def project(self, image_embedding) -> torch.TensorType:
        if not isinstance(image_embedding, torch.Tensor):
            print(f"the type of image_embedding is '{type(image_embedding)}' converting it to torch tensor")
            image_embedding = torch.tensor(image_embedding, dtype=torch.float).to(self.device)
        
        orig_device = image_embedding.device
        
        if image_embedding.device != self.device:
            image_embedding = image_embedding.to(self.device)
        

        image_embedding /= image_embedding.norm(dim=-1,keepdim=True)
        sim = image_embedding@self.embs_dataset.T.float()
        sim = (sim*100).softmax(dim=-1)
        prefix_embedding = sim@self.embs_dataset.float()
        prefix_embedding /= prefix_embedding.norm(dim=-1,keepdim=True)
        return prefix_embedding.to(orig_device)
    
    def _load_support_memory(self) -> Tuple[np.ndarray, np.ndarray]:

        if not os.path.exists(self.H5PY_FILE_PATH):
            print(f"[-] _load_support_memory: the path '{self.H5PY_FILE_PATH}' does not exist [-]")
            return None, None

        with h5py.File(self.H5PY_FILE_PATH, 'a') as hf:

            if self.H5PY_EMBEDDINGS_DATASET_NAME in hf:
                embeddings_dataset = hf[self.H5PY_EMBEDDINGS_DATASET_NAME][:]
                text_dataset = hf[self.H5PY_TEXT_DATASET_NAME][:]
            else:
                embeddings_dataset = None
                text_dataset = None
        
        return embeddings_dataset, text_dataset


        
    def _build_support_memory(self) -> Tuple[np.ndarray, np.ndarray]:
        ## construct the support memory

        self._load_models()

        if self.type == ProjectionType.COCO_CAPTIONS:
            from pycocotools.coco import COCO
            coco_obj = COCO(Im2TxtProjector.ANNOTATIONS_CAPTION_FILE_PATH)
            data = random.sample(list(coco_obj.anns.values()), k=Im2TxtProjector.SUPPORT_MEMORY_SIZE)
            data = [ d['caption'] for d in data ]
        elif self.type == ProjectionType.MS_MARCO_QUERIES_A:
            print(f"Loading MSMarco queries from file ", Im2TxtProjector.MS_MARCO_QUERIES_FILE_PATH)
            with open(Im2TxtProjector.MS_MARCO_QUERIES_FILE_PATH, "r") as input_file:
                lines = input_file.readlines()
            data = random.sample(lines, k=Im2TxtProjector.SUPPORT_MEMORY_SIZE)
            data = [ d.split("\t")[1].replace("\n", "") for d in data ]
            print(f"Loaded from file '{Im2TxtProjector.SUPPORT_MEMORY_SIZE}' lines, example of line: '{data[0]}'")
        else:
            #data = random.sample(data,500000)
            print(f"[!] Unimplemented data type '{self.type}'[!]")
            return None, None
        
        text_features = []
        captions = []
        batch_size = 1000
        self.clip_model.eval()
        for i in tqdm(range(0,len(data[:])//batch_size)):
            
            texts = data[i*batch_size:(i+1)*batch_size]
            with torch.no_grad():
                texts_token = self.tokenizer(texts).to(self.device)
                text_feature = self.clip_model.encode_text(texts_token)
                text_features.append(text_feature)
                captions.extend(texts)

        text_features = torch.cat(text_features,dim=0)
        text_features /= text_features.norm(dim=-1,keepdim=True).float()
        
        # store captions and text features in hdf5 dataset

        text_features_ndarray = text_features.cpu().numpy()

        assert len(text_features_ndarray) == len(captions), f"len(text_features_ndarray) = {len(text_features_ndarray)} != len(captions) = {len(captions)}"

        #if not os.path.exists(self.H5PY_FILE_PATH):
        #    print(f"os.path '{self.H5PY_FILE_PATH}' does not exists")
            

        with h5py.File(self.H5PY_FILE_PATH, 'w') as hf:

            if self.H5PY_EMBEDDINGS_DATASET_NAME in hf:
                embeddings_dataset = hf[self.H5PY_EMBEDDINGS_DATASET_NAME]
                text_dataset = hf[self.H5PY_TEXT_DATASET_NAME]
                print(f"[!] Dataset '{self.H5PY_EMBEDDINGS_DATASET_NAME}' already exists! Going to overwrite [!]")
            else:
                embeddings_dataset = hf.create_dataset(self.H5PY_EMBEDDINGS_DATASET_NAME, shape=(Im2TxtProjector.SUPPORT_MEMORY_SIZE, Im2TxtProjector.EMBEDDINGS_DIMENSION), dtype='float32')
                text_dataset = hf.create_dataset(self.H5PY_TEXT_DATASET_NAME, shape=(Im2TxtProjector.SUPPORT_MEMORY_SIZE, ), dtype=h5py.string_dtype(encoding='utf-8'))    #, dtype='str'
        
            for num_row in range(len(text_features_ndarray)):
                embeddings_dataset[num_row] = text_features_ndarray[num_row]
                text_dataset[num_row] = captions[num_row]

        return embeddings_dataset, text_dataset

    clip_model = None
    def _load_models(self):

        if self.clip_model is not None:
            # case already done
            return

        import clip

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.tokenizer = clip.tokenize
