
from recaption_model_vqa import blip_recaption, bert_recaption
from recap_dataset import Coco_Dataset, Coco_Dataset_things, Coco_Dataset_attributes, Coco_Dataset_Ref, DOCCI, Coco_Dataset_VQA, Coco_Dataset_dummy, FineCapEval, DCI
from torch.utils.data import DataLoader, Subset
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from accelerate import Accelerator
import numpy as np
import json
from collections import defaultdict
from pycocoevalcap.cider.cider import Cider
import itertools
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer
from torch.nn import DataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
import warnings
# warnings.filterwarnings("ignore")
from distinct_n import distinct_n_sentence_level
import re
import string
import collections
import argparse

# from recaption_model import blip_recaption, bert_recaption
from recap_dataset import Coco_Dataset, Coco_Dataset_things, Coco_Dataset_attributes, Coco_Dataset_Ref, Coco_Dataset_VQA
from torch.utils.data import DataLoader, Subset
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm
from accelerate import Accelerator
import numpy as np
import os
import json
from collections import defaultdict
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import itertools
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer, RobertaModel
from torch.nn import DataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
import evaluate


import pandas as pd

def load_references_from_json_finecap(file_path=None):

    data = pd.read_csv("./FineCapEval/FineCapEval.csv")

    
    captions_by_id = defaultdict(list)
    for i in range (len(data)):
        captions_by_id[data.iloc[i]['image']].append(data.iloc[i]['caption'])

    
    return captions_by_id
def load_references_from_json_dci(file_path=None):

    data = pd.read_csv("./FineCapEval/FineCapEval.csv")
    with open("./dci/densely_captioned_images/splits.json") as f:
        data = json.load(f)
    data = data["train"]+data["valid"]+data["test"]
    
    captions_by_id = defaultdict(list)
    for i in range (len(data)):
        cap_id = data[i]
        path = "./dci/densely_captioned_images/complete/"
        cap_path = os.path.join(path,cap_id)
        with open(cap_path) as f:
            cap_dict = json.load(f)

        captions = cap_dict["summaries"]["base"]
        img_id = cap_dict["image"]

        # span_start = self.down.iloc[i]["start_span"]
        # span_end = self.down.iloc[i]["end_span"]
        # images_path = "/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/Recaption/dci/densely_captioned_images/photos"
        # path = os.path.join(images_path,img_id)
        captions_by_id[img_id] = captions

    return captions_by_id
def load_references_from_json_DOCCI(file_path=None):

    data = pd.read_json("./docci/docci_descriptions.jsonlines", lines=True)
    data = data[data["split"]=="test"]

    
    captions_by_id = defaultdict(list)
    for i in range (len(data)):
        captions_by_id[data.iloc[i]['image_file']].append(data.iloc[i]['description'])

    
    return captions_by_id

def load_references_from_json_val(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    captions_by_id = defaultdict(list)
    data = data["annotations"]

    data = pd.DataFrame(data)
      
    data = data.sample(n=1500,  random_state=42)

    data = data.to_json(orient='records')
    data = json.loads(data)
    for item in data:
        captions_by_id[item['image_id']].append(item['caption'])
    
    return captions_by_id

def evaluate_cider(ref_captions, gen_captions):
    scorer = Cider()
    cider_score, _ = scorer.compute_score(gen_captions, ref_captions)
    # print(f"CIDEr Score: {cider_score}")
    return cider_score
def evaluate_rouge(ref_captions, gen_captions):
    scorer = Rouge()
    rouge_score, _, recall = scorer.compute_score(gen_captions, ref_captions)
    # print(f"rouge Score: {rouge_score}")
    return rouge_score, recall



# ref_caps = load_references_from_json("./my_datasets/captions_val2014.json")
def evaluation(model, dataloader_val, dataset):
    device = "cuda"
    if dataset == "fine":
        ref_caps = load_references_from_json_finecap()  
    if dataset =="dci":
        ref_caps = load_references_from_json_dci()
    if dataset =="doc":
        ref_caps = load_references_from_json_DOCCI()

    model.eval()
    # model = DDP(model.module)
    # model = model.to(device)
    gen_caps_contrastive = {}
    gen_caps_beam = {}
    with tqdm( unit='it', total=len(dataloader_val)) as pbar:

        for it, (_,_, captions,_,images, prompts, ids) in enumerate(dataloader_val):
            images, prompts = images.to(device), prompts.to(device)

            with torch.no_grad():
                # output = model.module.generate(images,captions,prompts, ids)
                processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl-coco")
                # generated_ids = model.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=60,num_beams= 5)
                generated_ids_contrastive  = model.model_blip.generate(input_ids=prompts, pixel_values=images,penalty_alpha=0.6, top_k=5,  max_length=60 )
                generated_ids_beam = model.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=60,num_beams= 5, no_repeat_ngram_size=2, repetition_penalty=1.5)
                # generated_ids = model.model.generate(input_ids=prompts, pixel_values=images, max_length=60)
                generated_caption = processor.batch_decode(generated_ids_contrastive, skip_special_tokens=True)

                output_contrastive = {key: [value] for key,value in zip(ids[0], generated_caption)}
                generated_caption = processor.batch_decode(generated_ids_beam, skip_special_tokens=True)
                output_beam = {key: [value] for key,value in zip(ids[0], generated_caption)}
            gen_caps_contrastive = {**gen_caps_contrastive, **output_contrastive}
            gen_caps_beam = {**gen_caps_beam, **output_beam}

            pbar.update()
    print(gen_caps_contrastive)
    print(gen_caps_beam)
    score_contrastive_c = evaluate_cider(gen_caps_contrastive, ref_caps)
    score_beam_c = evaluate_cider(gen_caps_beam, ref_caps)

    score_beam_c = evaluate_cider(gen_caps_beam, ref_caps)
    score_contrastive_r, recall_c = evaluate_rouge(gen_caps_contrastive, ref_caps)
    score_beam_r,recall = evaluate_rouge(gen_caps_beam, ref_caps)
    print(f"Recall contrastive: {recall_c.mean():.2f} ")
    print(f"Recall beam: {recall.mean():.2f} ")
    print(f"CIDEr contrastive: {score_contrastive_c:.2f} ")
    print(f"CIDEr beam: {score_beam_c:.2f} ")

    with open("captions "+str(dataset)+"beam.jsonl", "w") as f:
        for item in gen_caps_beam:
            f.write(json.dumps(item) + "\n")
    with open("captions "+str(dataset)+"contrastive.jsonl", "w") as f:
        for item in gen_caps_contrastive:
            f.write(json.dumps(item) + "\n")
    return score_contrastive_c, score_beam_c, score_contrastive_r, score_beam_r, recall_c.mean()
    # return score_contrastive_c, score_beam_c, score_contrastive_r, score_beam_r, recall_c.mean()


import wandb
import os 
os.environ["WANDB_API_KEY"] = "ee6091224cb7bb0fda72ab4cd492e55463c4813b"
with wandb.init(project="recap",  name="fine_T5_final"):

    model_captioning = blip_recaption()
    model_captioning = model_captioning.to("cuda")
    Fine = FineCapEval(split="val", downstream = False)
    DCI_D = DCI(split = "val", downstream= False)
    DOC = DOCCI(split = "val", downstream= False)

    Fine_L = DataLoader(Fine, batch_size=32,shuffle=False, num_workers=1,collate_fn = Fine.collate_fn,
                                        drop_last=False)
    DCI_L = DataLoader(DCI_D, batch_size=32,shuffle=False, num_workers=1,collate_fn = DCI_D.collate_fn,
                                        drop_last=False)
    DOC_L = DataLoader(DOC, batch_size=32,shuffle=False, num_workers=1,collate_fn = DOC.collate_fn,
                                        drop_last=False)
    trained_models = [              
                                 "eos_delay",   






                                
    ]

    for mod in trained_models:
        try:
            model_captioning_state_dict = torch.load(mod, map_location='cpu')
            model_captioning.load_state_dict(model_captioning_state_dict, strict=True)

        except: 
            print("error loading model")
        print("######################################################################")
        print(mod)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$DOC$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive = evaluation(model_captioning,DOC_L, "doc" )
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$Fine$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive = evaluation(model_captioning,Fine_L, "fine" )
        print("$$$$$$$$$$$$$$$$$$$$$$$$$DCI$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive = evaluation(model_captioning,DCI_L, "dci" )