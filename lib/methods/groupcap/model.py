from enum import Enum
from typing import Callable, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import numpy as np
from typing import List

import json, os

from lib.sorting import dissimilarity_ranking
#from clipcap.clipcap import get_generated_captions

class LLMTypes(Enum):
    Gemma7b_IT     = 'Gemma7b_IT'
    Gemma2b_IT     = 'Gemma2b_IT'
    Mistral7b_IT   = 'Mistral7b_IT'
    LLama2_7b   = 'LLama2_7b'
    LLama3_8b   = 'LLama3_8b'
    LLama3_8b_IT   = 'LLama3_8b_IT'

class ImageCaptioningMethods(Enum):
    decap       = 'decap'
    clipcap     = 'clipcap'

class LLMBasePrompts(Enum):
    default = "write one single short phrase that contains only information valid for all the given phrases: \n"
    few_shot = ""

class LLMQEPrompts(Enum):
    default = """Please, I need your help to rewrite this original query for a search engine: '{original_query}'.\n\n
    The query that I want you to generate must be similar to the original one, and must contain only information valid for most of the given phrases:\n 
    {phrases}\n\n
    Please answer with the generated query for the search engine and nothing more. Do not write any delimiter, write only the generated query. 
    The generated query must be short. The generated query must be an expansion of the original query, and must be made of one single phrase"""
    ver1 = """   Starting from the following original query write another query that is also based on the content of the following phrases. The query that you will build must be short, similar to the original one and must contain only information valid for all the given phrases: 
                    {phrases}
                    [ORIGINAL QUERY] {original_query}"""

class GroupCapLLM:

    tokenizer = {}
    model = {}

    def __init__(self, image_captioning_type: ImageCaptioningMethods, image_captioning_method : Callable[[torch.TensorType], List[str]], 
                 llm: LLMTypes, device_map: str = 'cpu', 
                 image_captioning_method_device : str = None, llm_prompt: LLMBasePrompts = LLMBasePrompts.default, 
                 use_top_k: int = 4,
                 use_pipeline : bool = True) -> None:
        """
        you can specify the captioning method to use, the large language model to adopt to write the group caption 
        and the amount of images to use as basis of the generation
        - llm_prompt : str
        - use_top_k : int
        """
        if not isinstance(llm, LLMTypes):
            print(f"Invalid parameter {llm = } that is not of LLMTypes type [!]")
            return
        if image_captioning_type is None:
            print(f"{image_captioning_type = }! cannot use methods that need captioning method calls")
        elif not isinstance(image_captioning_type, ImageCaptioningMethods):
            print(f"Invalid parameter {image_captioning_type = } that is not of ImageCaptioningMethods type [!]")
            return
        self.image_captioning_type = image_captioning_type
        self.image_captioning_method = image_captioning_method
        self.use_top_k = use_top_k
        self.device_map = device_map
        self.image_captioning_method_device = image_captioning_method_device if image_captioning_method_device is not None else self.device_map
        self.use_pipeline = use_pipeline
        self.set_prompt(llm_prompt)
        self._load_llm(llm, self.device_map)

        pass

    debug = False

    def set_debug(self, debug=True):
        self.debug = debug

    def generate_caption(self, cluster_embeddings) -> List[str]:
        """
        method that receives in input the CLIP latent space image embeddings
        and returns a caption that is descriptive for the whole group of images
        """
        if self.image_captioning_type is None:
            print(f"{self.image_captioning_type = }! cannot use methods that need captioning method calls")
            return ["GROUPCAP_0x14"]
        #print(f"{type(cluster_embeddings) = }")
        top_embs = self.get_most_significant_embeddings(cluster_embeddings)
        
        #if self.image_captioning_type == ImageCaptioningMethods.decap:
        #print(f"{type(top_embs) = } \t|\t{self.image_captioning_method_device = }")
        top_embs_on_device = torch.from_numpy(top_embs).to(self.image_captioning_method_device)
        if self.debug: print(f"Going to call the method {self.image_captioning_method} for the top embeddings that are on device {self.image_captioning_method_device}")
        captions = self.image_captioning_method(top_embs_on_device)
        #if verbose: print(f"Gemerated captions for top {self.use_top_k} represetnative embeddings: ", captions)
        
        return self.generate_caption_from_images_captions(captions)
    
    def generate_caption_from_images_captions(self, images_captions : List[str], prompt : str = None) -> List[str]:
        """
        - handle both the case in which that input prompt has {phrases} or the case in which it hasn't
        """
        if prompt is None:
            prompt = self.base_prompt
        tmp_str = ""
        for caption in images_captions:
            tmp_str += f"- {caption}\n"

        if "{phrases}" in prompt:
            prompt = prompt.format(phrases=tmp_str)
        else:
            prompt += tmp_str

        return self.__run_method(prompt)


    def __run_method(self, prompt):
        
        if self.debug: print(f"Generated prompt:\n", prompt)
        
        generated_answer = self.send_chat_message(prompt)

        return [ generated_answer ]
    
    def generate_expanded_query(self, cluster_embeddings, original_query : str) -> List[str]:
        """
        method that receives in input the CLIP latent space image embeddings
        and returns a caption that is descriptive for the whole group of images
        """
        if self.image_captioning_type is None:
            print(f"{self.image_captioning_type = }! cannot use methods that need captioning method calls")
            return ["GROUPCAP_0x14"]
        #print(f"{type(cluster_embeddings) = }")
        top_embs = self.get_most_significant_embeddings(cluster_embeddings)
        
        #if self.image_captioning_type == ImageCaptioningMethods.decap:
        #print(f"{type(top_embs) = } \t|\t{self.image_captioning_method_device = }")
        top_embs_on_device = torch.from_numpy(top_embs).to(self.image_captioning_method_device)
        if self.debug: print(f"Going to call the method {self.image_captioning_method} for the top embeddings that are on device {self.image_captioning_method_device}")
        captions = self.image_captioning_method(top_embs_on_device)
        #if verbose: print(f"Gemerated captions for top {self.use_top_k} represetnative embeddings: ", captions)
        return self.generate_expanded_query_from_images_captions(captions, original_query)
        
    def generate_expanded_query_from_images_captions(self, images_captions, original_query : str, qe_prompt : str = None) -> List[str]:
        """
        - the parameter qe_prompt must be either None (in that case the self.qe_prompt string is used) or a string with the following format:
        - example of qe_prompt string
        - qe_prompt = "Rewrite this original query '{original_query}' using the additional context that must be shared among all the following phrases: {phrases}"
        """
        captions = ""
        for caption in images_captions:
            captions += f"- {caption}\n"

        if qe_prompt is None:
            qe_prompt = self.qe_prompt

        prompt = qe_prompt.format(phrases=captions, original_query=original_query)
        
        return self.__run_method(prompt)


    def get_most_significant_embeddings(self, cluster_embeddings):
        
        most_repr = dissimilarity_ranking(cluster_embeddings)
        top_embs = cluster_embeddings[most_repr[:self.use_top_k]]
        return top_embs

    def send_chat_message(self, input_prompt : str) -> str:
        chat = [
            {"role": "user", "content": input_prompt},
        ]

        if self.use_pipeline:
            #print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])
            return self.pipeline(chat)[0]['generated_text'][-1]['content']

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt")

        outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=150)
        generated_chat = self.tokenizer.decode(outputs[0])
        new_ret = generated_chat.replace("<s>", "").replace("</s>", "").replace(prompt.replace("<s>", "").replace("</s>", ""), "")
        
        tokens_to_remove = ["[INST]", "[/INST]"]
        
        for token in tokens_to_remove:
            new_ret = new_ret.replace(token, "")
        
        new_ret = new_ret.strip()
        if new_ret.startswith("\""): new_ret = new_ret[1:]
        if new_ret.endswith("\""): new_ret = new_ret[:-1]
        return new_ret

    def set_prompt(self, base_prompt):
        
        self.qe_prompt = LLMQEPrompts.default.value

        if not isinstance(base_prompt, LLMBasePrompts) or len(base_prompt.value) == 0:
            self.base_prompt = LLMBasePrompts.default.value
            print(f"Using default base prompt { self.base_prompt = } since the given prompt is not valid ({base_prompt = })")
            return
        
        self.base_prompt = base_prompt.value

    def _load_llm(self, llm_type : LLMTypes, device_map : str):
        """
        loads the llm object.

        It downloads it too eventually
        """
        dtype=torch.bfloat16
        quantization_config=None

        if llm_type == LLMTypes.Gemma7b_IT:
            model_id = "google/gemma-7b-it"
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            dtype=torch.int8
        elif llm_type == LLMTypes.Gemma2b_IT:
            model_id = "google/gemma-2b-it"
        elif llm_type == LLMTypes.Mistral7b_IT:
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            dtype=None #torch.bfloat16
        elif llm_type == LLMTypes.LLama3_8b:
            model_id = "meta-llama/Meta-Llama-3-8B"
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            dtype=None #torch.bfloat16
        elif llm_type == LLMTypes.LLama3_8b_IT:
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            dtype=None #torch.bfloat16
        else:
            #case default
            model_id = "google/gemma-7b-it"
            print(f"Undefined model for { llm_type = } | using default { model_id = }")

        if not ( hasattr(GroupCapLLM, f"model_{model_id}") and hasattr(GroupCapLLM, f"tokenizer_{model_id}") ):
            setattr(GroupCapLLM, f"tokenizer_{model_id}", AutoTokenizer.from_pretrained(model_id) )
            setattr(GroupCapLLM, f"model_{model_id}", AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=dtype,
                quantization_config=quantization_config
            ) )
        self.tokenizer = getattr(GroupCapLLM, f"tokenizer_{model_id}")
        self.model = getattr(GroupCapLLM, f"model_{model_id}")

        if self.use_pipeline:
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        else:
            self.pipeline = None

    def get_prompts_dict() -> dict:
        """
        loads from file the dictionary of the available group captioning prompts
        """
        with open(os.path.join( os.path.dirname(__file__), 'prompts.json'), 'r') as input_file:
            prompts_dict = json.load(input_file)
        return prompts_dict
    
    def add_to_prompts_dict(prompt_type : str, prompt_name : str, prompt : str) -> dict:
        """
        returns the updated dictionary on success
        - the prompt must contain '{phrases}' and also '{original_query}' if its type is 'query-expansion'
        - the allowed prompt_type values are the keys of GroupCapLLM.get_prompts_dict() (should be 'query-expansion' and 'group-captioning')
        - if the given prompt_name already exists, it overwrites the prompt with that name
        """
        current_dict = GroupCapLLM.get_prompts_dict()
        assert prompt_type in current_dict.keys(), f"{prompt_type = } not in {current_dict.keys() = }"
        assert "{phrases}" in prompt, f"{{phrases}} not in {prompt = }"
        if prompt_type == 'query-expansion':
            assert "{original_query}" in prompt, f"{{original_query}} not in {prompt = }"
        if prompt_name in current_dict[prompt_type].keys():
            print(f"[add_to_prompts_dict] {prompt_name = } is already present, overwriting ... ")
        current_dict[prompt_type][prompt_name] = prompt
        
        with open(os.path.join( os.path.dirname(__file__), 'prompts.json'), 'w') as output_file:
            json.dump(current_dict, output_file)
        return current_dict

