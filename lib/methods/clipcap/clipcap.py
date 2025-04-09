import os

from dotenv import load_dotenv

load_dotenv()

if os.getenv("CLIPCAP_ENABLED").lower() not in ["0", "", "no", "false"]:

    try:
        from lib.methods.clipcap.model import *
    except Exception as e:
        print("Exception occurred while doing 'from clipcap.model import *' ", e)
        from model import *

    working_directory = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(working_directory, "pretrained_models")
    os.makedirs(save_path, exist_ok=True)
    #model_path = os.path.join(save_path, 'model_weights.pt')

    #@title Choose pretrained model - COCO or Coneptual captions

    try:
        from lib.methods.clipcap.downloader import downloader
    except Exception as e:
        print("Exception occurred while doing 'from clipcap.downloader import downloader' ", e)
        from downloader import downloader

    pretrained_model = 'COCO'#'Conceptual captions'  # @param ['COCO', 'Conceptual captions']

    if pretrained_model == 'Conceptual captions':
        model_path = os.path.join(save_path, 'conceptual-captions_model_weights.pt')
        if not os.path.exists(model_path):
            downloader.download_file("14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT", model_path)
    else:
        model_path = os.path.join(save_path, 'COCO_model_weights.pt')
        if not os.path.exists(model_path):
            downloader.download_file("1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX", model_path)

    #@title CLIP model + GPT2 tokenizer

    is_gpu = True

    device = CUDA(0) if is_gpu else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    #@title Load model weights


    prefix_length = 10

    clipcap_model = ClipCaptionModel(prefix_length)

    # map_location was CPU, now I set device
    altered_state_dict = torch.load(model_path, map_location=CPU)
    for i in range(12):
        del altered_state_dict['gpt.transformer.h.' + str(i) + '.attn.bias']
        del altered_state_dict['gpt.transformer.h.' + str(i) + '.attn.masked_bias']
    # https://github.com/rmokady/CLIP_prefix_caption/issues/76
    clipcap_model.load_state_dict(altered_state_dict)
    #clipcap_model.load_state_dict(torch.load(model_path, map_location=CPU), strict=False) #, strict=False

    clipcap_model = clipcap_model.eval() 
    device = CUDA(0) if is_gpu else "cpu"
    clipcap_model = clipcap_model.to(device)

    def get_generated_captions(embeddings_on_device, use_beam_search=False):

        ret_list = []

        with torch.no_grad():
            for embedding in embeddings_on_device:
                #prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                prefix_embed = clipcap_model.clip_project(embedding).reshape(1, prefix_length, -1)
                #import hashlib
                #embedding_digest = hashlib.sha256( embedding.cpu().numpy().data ).hexdigest()
                #prefix_embed_digest = hashlib.sha256( prefix_embed.cpu().numpy().data ).hexdigest()
                #print(F"[get_generated_captions] {prefix_embed.shape = } - {prefix_embed_digest = } - {embedding.shape = } - {embedding_digest = }")
                    
                if use_beam_search:
                    generated_text_prefix = generate_beam(clipcap_model, gpt2_tokenizer, embed=prefix_embed)[0]
                else:
                    generated_text_prefix = generate2(clipcap_model, gpt2_tokenizer, embed=prefix_embed)


                #print('\n')
                #print(generated_text_prefix)
                if type(generated_text_prefix) == list:
                    ret_list.append(generated_text_prefix[0])
                else:
                    ret_list.append(generated_text_prefix)

        return ret_list

    def get_generated_captions_query_expansion(embeddings_on_device, initial_query="", use_beam_search=False):

        #ret_list = []

        with torch.no_grad():
            prefix_embed = clipcap_model.clip_project(embeddings_on_device).reshape(len(embeddings_on_device), prefix_length, -1)
            
            # get token representation of initial_query
            tokenized_input_ids = gpt2_tokenizer.encode(initial_query, return_tensors="pt").to(device)
            initial_query_tokenized = clipcap_model.gpt.transformer.wte(tokenized_input_ids)
            
            initial_query_tokenized.reshape(1, -1, prefix_embed.shape[2])

            # stack initial query such that has the same shape for axis 0 and axis 2 of 
            # the tokenized image embedding obtained from the clipcap model
            tokenized_input_stacked = np.tile(initial_query_tokenized.to(CPU), (prefix_embed.shape[0], 1, 1))

            # Concatenate tokenized_input_stacked and prefix_embed along the first axis
            combined_input_embeddings = torch.from_numpy(
                                            np.concatenate((tokenized_input_stacked, prefix_embed.to(CPU)), axis=1)
                                        ).to(device)

            #print(combined_input_embeddings.shape)  # Shape: (10, 15, 768)
                
            if use_beam_search:
                generated_text_prefix = generate_beam(clipcap_model, gpt2_tokenizer, embed=combined_input_embeddings)[0]
            else:
                generated_text_prefix = generate2(clipcap_model, gpt2_tokenizer, embed=combined_input_embeddings, entry_count=len(combined_input_embeddings))


            #print('\n')
            #print(generated_text_prefix)
                
            #ret_list.append(generated_text_prefix)
        #print("type of generated_text_prefix", type(generated_text_prefix))
        #print("generated_text_prefix[0]", generated_text_prefix[0])
        #print("type of generated_text_prefix[0]", type(generated_text_prefix[0]) )
        return generated_text_prefix


    def get_generated_caption_query_expansion_on_top_cluster_items(embeddings_on_device, initial_query="", use_beam_search=False):
        """
        Receives as input a set of embeddings representatives of *one* cluster and a prompt, 
        then returns the generated expanded query for that cluster.
        
        - receives as input embeddings_on_device whose shape must be (X, CLIP-image-embeddings-dimensionality)

        where X is the number of embeddings, while CLIP-image-embeddings-dimensionality is usually 512.


        """
        #ret_list = []

        assert embeddings_on_device.device == device, f"{embeddings_on_device.device = } == {device = }"

        with torch.no_grad():
            prefix_embed = clipcap_model.clip_project(embeddings_on_device).reshape(
                    1, prefix_length * len(embeddings_on_device), -1
                    )
            
            # get token representation of initial_query
            if not ( initial_query == "" or initial_query is None ):
                tokenized_input_ids = gpt2_tokenizer.encode(initial_query, return_tensors="pt").to(device)
                initial_query_tokenized = clipcap_model.gpt.transformer.wte(tokenized_input_ids)
                
                initial_query_tokenized.reshape(1, -1, prefix_embed.shape[2])

                # stack initial query such that has the same shape for axis 0 and axis 2 of 
                # the tokenized image embedding obtained from the clipcap model
                #tokenized_input_stacked = np.tile(initial_query_tokenized.to(CPU), (prefix_embed.shape[0], 1, 1))

                # Concatenate tokenized_input_stacked and prefix_embed along the first axis
                combined_input_embeddings = torch.concatenate(
                                                (initial_query_tokenized.to(device), prefix_embed.to(device)), axis=1
                                            ).to(device)
            else:
                combined_input_embeddings = prefix_embed.to(device)

            #print("combined_input_embeddings.shape: ", combined_input_embeddings.shape)  # Shape: (10, 15, 768)
                
            if use_beam_search:
                generated_text_prefix = generate_beam(clipcap_model, gpt2_tokenizer, embed=combined_input_embeddings)[0]
            else:
                generated_text_prefix = generate2(clipcap_model, gpt2_tokenizer, embed=combined_input_embeddings, entry_count=len(combined_input_embeddings))


            #print('\n')
            #print(generated_text_prefix)
                
            #ret_list.append(generated_text_prefix)
        #print("type of generated_text_prefix", type(generated_text_prefix))
        #print("generated_text_prefix[0]", generated_text_prefix[0])
        #print("type of generated_text_prefix[0]", type(generated_text_prefix[0]) )
        #print("-------------------")
        #print("going to return : ", generated_text_prefix)
        #print("-------------------")
        return generated_text_prefix
else:
    def get_generated_captions(embeddings_on_device, use_beam_search=False):
        return ["CLIPCAP is disabled"] * len(embeddings_on_device)
    
    def get_generated_captions_query_expansion(embeddings_on_device, initial_query="", use_beam_search=False):
        return ["CLIPCAP is disabled"] * len(embeddings_on_device)
    
    def get_generated_caption_query_expansion_on_top_cluster_items(embeddings_on_device, initial_query="", use_beam_search=False):
        #print("-------------------")
        #print("-------------------")
        return ["CLIPCAP is disabled"]