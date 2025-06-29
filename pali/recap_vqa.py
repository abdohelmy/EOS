from recaption_model_vqa import blip_recaption, bert_recaption
from recap_dataset import Coco_Dataset, Coco_Dataset_things, Coco_Dataset_attributes, Coco_Dataset_Ref, Coco_Dataset_VQA, Coco_Dataset_dummy, FineCapEval, DOCCI, DCI
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
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer, LlavaForConditionalGeneration, AutoModelForCausalLM
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
from distinct_n import distinct_n_corpus_level
from distinct_n import distinct_n_sentence_level
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
from nltk.stem import *
from nltk.stem.porter import *
import fasttext
import fasttext.util
class MultiClassBCELoss(nn.Module):
    def __init__(self,
                 use_weight_mask=False,
                 use_focal_weights=False,
                 focus_param=2,
                 balance_param=0.25
                 ):
        super().__init__()

        self.use_weight_mask = use_weight_mask
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.use_focal_weights = use_focal_weights
        self.focus_param = focus_param
        self.balance_param = balance_param
        
    def forward(self,
                outputs,
                targets,
                weights):
        # inputs and targets are assumed to be BatchxClasses
        assert len(outputs.shape) == len(targets.shape)
        assert outputs.size(0) == targets.size(0)
        assert outputs.size(1) == targets.size(1)
        
        # weights are assumed to be BatchxClasses
        assert outputs.size(0) == weights.size(0)
        assert outputs.size(1) == weights.size(1)

        if self.use_weight_mask:
            bce_loss = F.binary_cross_entropy_with_logits(input=outputs,
                                                          target=targets,
                                                          weight=weights)            
        else:
            bce_loss = self.nll_loss(input=outputs,
                                     target=targets)
        
        if self.use_focal_weights:
            logpt = - bce_loss
            pt    = torch.exp(logpt)

            focal_loss = -((1 - pt) ** self.focus_param) * logpt
            balanced_focal_loss = self.balance_param * focal_loss
            
            return balanced_focal_loss
        else:
            return bce_loss 

# coco_train = Coco_Dataset()
# coco_val = Coco_Dataset(split = "val")
stuff_objects_train = Coco_Dataset_things()
stuff_objects_coco_train = Coco_Dataset_things(downstream= False)#with image
stuff_objects_coco_val = Coco_Dataset_things(split="val", downstream= False)#with image
stuff_objects_val = Coco_Dataset_things(split = "val")

vqa_train = Coco_Dataset_VQA()
vqa_val = Coco_Dataset_VQA(split = "val")
# # Create shuffled indices
# indices = torch.randperm(len(stuff_objects_train)).tolist()

# # Create subsets from the datasets
# coco_train_shuffeled = Subset(coco_train, indices)
# stuff_objects_train_shuffeled = Subset(stuff_objects_train, indices)

# model_blip = blip_recaption()
model_down = bert_recaption()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_blip = model_blip.to(device)
# model_blip = DDP(model_blip)

# model_down = model_down.to(device)
# model_down = DDP(model_down)
for name, param in model_down.named_parameters():
        param.requires_grad = True

# for name, param in model_down.module.named_parameters():
#     if "model" in name or "Bert" in name and "vqa" not in name and "pooler" not in name:
#         param.requires_grad = False

# for name, param in model_down.module.model.encoder.layer[-1].named_parameters():
#     param.requires_grad = True
# for name, param in model_down.module.model.encoder.layer[-2].named_parameters():
#     param.requires_grad = True
# for name, param in model_down.module.model.encoder.layer[-3].named_parameters():
#     param.requires_grad = True
# for name, param in model_down.module.vqa.named_parameters():
#     param.requires_grad = True

# optimizer1 = torch.optim.AdamW(model_blip.module.parameters(), lr=5e-5)
# optimizer2 = torch.optim.SGD(model_down.module.parameters(), lr=0.0001)
# params = list(model_blip.parameters()) + list(model_down.parameters())
optimizer_pretraining = torch.optim.AdamW(model_down.parameters(), lr=5e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pretraining, 'min', patience = 5, factor = 0.5)
# optimizer3 = torch.optim.Adam(params, lr=0.0001)
device = "cuda" if torch.cuda.is_available() else "cpu"
#coco_dataloader_train = DataLoader(coco_train, batch_size=4,shuffle=True, collate_fn = coco_train.collate_fn, num_workers=1,
#                                    drop_last=True)
#coco_dataloader_val = DataLoader(coco_val, batch_size=32,shuffle=False, collate_fn = coco_train.collate_fn, num_workers=1,
#                                    drop_last=True)

things_dataloader_train = DataLoader(stuff_objects_train, batch_size=512,shuffle=True, num_workers=4,collate_fn = stuff_objects_train.collate_fn,
                                    drop_last=True) #no image
things_coco_dataloader_train = DataLoader(stuff_objects_coco_train, batch_size=12,shuffle=True, num_workers=4,collate_fn = stuff_objects_coco_train.collate_fn,
                                    drop_last=True) #with image
things_coco_dataloader_val = DataLoader(stuff_objects_coco_val, batch_size=24,shuffle=False, num_workers=4,collate_fn = stuff_objects_coco_val.collate_fn,
                                    drop_last=True)

vqa_dataloader_train = DataLoader(vqa_train, batch_size=256,shuffle=True,collate_fn = vqa_train.collate_fn,
                                    drop_last=True) #with image
vqa_dataloader_val = DataLoader(vqa_val, batch_size=128,shuffle=False, collate_fn = vqa_val.collate_fn,
                                    drop_last=True)
# model, optimizer, dataloader_train = accelerator.prepare(
#      model, optimizer, dataloader_train
#  )

# 



def load_references_from_json(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    captions_by_id = defaultdict(list)
    for item in data["annotations"]:
        captions_by_id[item['image_id']].append(item['caption'])
    
    return captions_by_id
import pandas as pd
def load_references_from_json_finecap(file_path=None):

    data = pd.read_csv("/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/FineCapEval/FineCapEval.csv")

    
    captions_by_id = defaultdict(list)
    for i in range (len(data)):
        captions_by_id[data.iloc[i]['image']].append(data.iloc[i]['caption'])

    
    return captions_by_id
def load_references_from_json_dci(file_path=None):    
    with open("/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/Recaption/dci/densely_captioned_images/splits.json") as f:
        data = json.load(f)
    data = data["train"]+data["valid"]+data["test"]    
    captions_by_id = defaultdict(list)
    for i in range (len(data)):
        cap_id = data[i]
        path = "/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/Recaption/dci/densely_captioned_images/complete/"
        cap_path = os.path.join(path,cap_id)
        with open(cap_path) as f:
            cap_dict = json.load(f)        
        captions = cap_dict["summaries"]["base"]
        img_id = cap_dict["image"]        # span_start = self.down.iloc[i]["start_span"]
        # span_end = self.down.iloc[i]["end_span"]
        # images_path = "/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/Recaption/dci/densely_captioned_images/photos"
        # path = os.path.join(images_path,img_id)
        captions_by_id[img_id] = captions    
    return captions_by_id
def load_references_from_json_DOCCI(file_path=None):

    data = pd.read_json("/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/Recaption/docci/docci_descriptions.jsonlines", lines=True)
    data = data[data["split"]=="test"]

    
    captions_by_id = defaultdict(list)
    for i in range (len(data)):
        captions_by_id[data.iloc[i]['image_file']].append(data.iloc[i]['description'])

    
    return captions_by_id
def evaluate_cider(gen_captions, ref_captions):
    scorer = Cider()
    cider_score, _ = scorer.compute_score(ref_captions, gen_captions)
    print(f"CIDEr Score: {cider_score}")
    return cider_score



ref_caps = load_references_from_json("./my_datasets/captions_val2014.json")
ref_caps = []
avg_len = 0
def evaluation(model, dataloader_val, ref_caps):
    # ref_caps = load_references_from_json_finecap()
    # ref_caps = load_references_from_json_dci()
    ref_caps = load_references_from_json("./my_datasets/captions_val2014.json")


    # ref_caps = load_references_from_json_DOCCI()
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
                processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
                # generated_ids = model.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=60,num_beams= 5)
                generated_ids_contrastive  = model.model_blip.generate(input_ids=prompts, pixel_values=images,penalty_alpha=0.6, top_k=5,  max_new_tokens=60 )[:,prompts.shape[1]:]
                generated_ids_beam = model.model_blip.generate(input_ids=prompts, pixel_values=images, max_new_tokens=60,num_beams= 5, no_repeat_ngram_size=2, repetition_penalty=1.5)[:,prompts.shape[1]:]
                # generated_ids = model.model.generate(input_ids=prompts, pixel_values=images, max_length=60)
                generated_caption = processor.batch_decode(generated_ids_contrastive, skip_special_tokens=True)

                output_contrastive = {key: [value] for key,value in zip(ids[0], generated_caption)}
                generated_caption = processor.batch_decode(generated_ids_beam, skip_special_tokens=True)
                output_beam = {key: [value] for key,value in zip(ids[0], generated_caption)}
                # breakpoint()
            gen_caps_contrastive = {**gen_caps_contrastive, **output_contrastive}
            gen_caps_beam = {**gen_caps_beam, **output_beam}

            pbar.update()
    print(output_contrastive)
    print(output_beam)
    score_contrastive_c = evaluate_cider(gen_caps_contrastive, ref_caps)
    score_beam_c = evaluate_cider(gen_caps_beam, ref_caps)
    score_contrastive_r, recall_c = evaluate_rouge(gen_caps_contrastive, ref_caps)
    score_beam_r,recall_b = evaluate_rouge(gen_caps_beam, ref_caps)
    CHAIR_data = [{"image_id": int(key), "caption": caption} for key, captions in gen_caps_contrastive.items() for caption in captions]
    CHAIR_data_beam = [{"image_id": int(key), "caption": caption} for key, captions in gen_caps_beam.items() for caption in captions]



    return score_contrastive_c, score_beam_c, score_contrastive_r, score_beam_r, recall_c.mean(), recall_b.mean(), CHAIR_data, CHAIR_data_beam
    # return score_contrastive_c, score_beam_c, score_contrastive_r, score_beam_r, recall_c.mean()
# 
    # return score_contrastive_c, score_beam_c, score_contrastive_r, score_beam_r, recall_c.mean(), CHAIR_data
    return score_contrastive_c, score_beam_c, score_contrastive_r, score_beam_r, recall_c.mean(), avg_len

def load_references_from_json_val(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    captions_by_id = defaultdict(list)
    data = data["annotations"]

    data = pd.DataFrame(data)
    data = data.drop_duplicates(subset="image_id", keep="first")
    data = data.sample(n=300,  random_state=42)

    data = data.to_json(orient='records')
    data = json.loads(data)
    for item in data:
        captions_by_id[item['image_id']].append(item['caption'])
    
    return captions_by_id
def val_evaluation(model, dataloader_val, ref_caps):
    model.eval()
    # model = DDP(model.module)
    # model = model.to(device)
    ref_caps = load_references_from_json_val("./my_datasets/captions_val2014.json")

    gen_caps_contrastive = {}
    gen_caps_beam = {}
    all_probs_rest = []
    all_probs = []
    distinct_ns = 0.0
    avg_len = 0.0
    with tqdm( unit='it', total=len(dataloader_val)) as pbar:

        for it, (_,_, captions,_,images, prompts, ids) in enumerate(dataloader_val):
            images, prompts = images.to(device), prompts.to(device)

            with torch.no_grad():
                # output = model.module.generate(images,captions,prompts, ids)
                processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")

                # generated_ids = model.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=60,num_beams= 5)
                generated_ids_contrastive  = model.model_blip.generate(input_ids=prompts, pixel_values=images, max_new_tokens=100, output_logits= True, return_dict_in_generate=True) #greedy
                generated_ids_beam = model.model_blip.generate(input_ids=prompts, pixel_values=images, max_new_tokens=50,num_beams= 5, no_repeat_ngram_size=2, repetition_penalty=1.5)
                logits = generated_ids_contrastive[1]

                logits = torch.stack(logits, dim=1)
                indices = generated_ids_contrastive[0].unsqueeze(-1)[:,prompts.shape[1]:]
                logits_prob = torch.nn.functional.log_softmax(logits, dim=-1)
                words_prob = torch.gather(logits_prob, dim=2, index=indices).squeeze(-1)
                eos_indices = torch.nonzero(indices.squeeze(-1) == 1, as_tuple=False)
                eos_probs = torch.exp(words_prob[eos_indices[:,0],eos_indices[:,1]]).detach().cpu()
                every_prob = torch.exp(words_prob).detach().cpu()
                mask_prob_1 = (indices.squeeze(-1) != 0).detach().cpu()
                mask_prob_2 =  (indices.squeeze(-1) != 1).detach().cpu()
                mask_prob = mask_prob_1 * mask_prob_2
                every_prob_mean = every_prob[mask_prob].mean()
                all_probs_rest.append(every_prob_mean)
                all_probs.append(eos_probs.mean().item())
                # generated_ids = model.model.generate(input_ids=prompts, pixel_values=images, max_length=60)
                generated_caption = processor.batch_decode(generated_ids_contrastive[0][:,prompts.shape[1]:], skip_special_tokens=True)
                output_contrastive = {key: [value] for key,value in zip(ids[0], generated_caption)}
                generated_caption = processor.batch_decode(generated_ids_beam[:,prompts.shape[1]:], skip_special_tokens=True)
                output_beam = {key: [value] for key,value in zip(ids[0], generated_caption)}
                # clean_cap = [ x.split("/n")[0] for x in generated_caption]
                seq_recap = [len(x.split(' ')) for x in generated_caption]
                avg_seq_recap = np.mean(seq_recap)
                avg_len +=avg_seq_recap
                distinct_ns +=  distinct_n_corpus_level(generated_caption,1)/4+distinct_n_corpus_level(generated_caption,2)/4+distinct_n_corpus_level(generated_caption,3)/4+distinct_n_corpus_level(generated_caption,4)/4
            gen_caps_contrastive = {**gen_caps_contrastive, **output_contrastive}
            gen_caps_beam = {**gen_caps_beam, **output_beam}

            pbar.update()
    # rouge_contrastive, recall_contrastive = evaluate_rouge(gen_caps_contrastive, ref_caps)

    score_contrastive_r, recall_c = evaluate_rouge(gen_caps_contrastive, ref_caps)
    model.train()

    # _, score_beam = evaluate_rouge(gen_caps_beam, ref_caps)

    return gen_caps_contrastive, gen_caps_beam, np.mean(all_probs), np.mean(all_probs_rest), avg_len/it, recall_c.mean(), distinct_ns/it
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  norm = normalize_answer(s).split()
#   stemmed = [stemmer.stem(word) for word  in norm]
  return norm

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    nearest_pred_toks = []
    nearest_pred_toks.extend(pred_toks)
    # for ans in pred_toks:
    #     if len(ans.split(" ")) < 2:
    #         neighbours = fastmodel.get_nearest_neighbors(ans)
    #         near_list = [i[1] for i in neighbours]
    #         nearest_pred_toks.extend(near_list)
    common = collections.Counter(gold_toks) & collections.Counter(nearest_pred_toks)
    # breakpoint()
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    #  
    return f1
from sklearn.metrics import f1_score
def load_references_from_json(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    captions_by_id = defaultdict(list)
    for item in data["annotations"]:
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
def evaluate_spice(gen_captions, ref_captions):

    os.environ['CORENLP'] = '/home/israfel.salazar/abdo/Recaption/stanford-corenlp-4.5.7'
    
    scorer = Spice()
    spice_score, _ = scorer.compute_score(ref_captions, gen_captions)
    print(f"SPICE Score: {spice_score}")
    return spice_score


# ref_caps = load_references_from_json("./my_datasets/annotations/captions_val2017.json")
ref_caps_train = load_references_from_json("./my_datasets/annotations/captions_train2017.json")




def compute_score_batch_downstream(outputs, label, tokenized_input, answers, tokenizer, tokenized_inputs_gold):
                # tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
                outputs = outputs.permute(0,2,1) # Permute for Cross Entropy loss
                pred = outputs.argmax(dim=1)
                f_scores = []
                answers_pred = []
                answers_target = []

                for prediction, target, input_tokens, gold_tokens in zip(pred,label, tokenized_input, tokenized_inputs_gold):
                    indices_target = [loc for loc, val in enumerate(target) if val == 1] #get all answers begninning of target
                    indices_pred =  [loc for loc, val in enumerate(prediction) if val == 1] #get all answers beginning of prediction
                    answers_t = []

                    for ind_target in indices_target:
                        answer_indicies = []
                        for index,i in enumerate(target[ind_target::]): #loop to get all inside answers indecies
                            if i.item() ==0:
                                break #end of answer
                            offseted_index = index + ind_target #add answer beginning offset to get inside index (rest of answer)
                            answer_indicies.append(offseted_index) 
                        answers_tokens = input_tokens[ind_target: offseted_index+1] #get answer
                        # for tok in answers_tokens:
                        ans_text = tokenizer.decode(answers_tokens, skip_special_tokens=True) # decode it
                        answers_t.append(ans_text) #all answers of one sample
                    answers_target.append(answers_t) #Not used, just collecting all answers
                    answers_p = []
                    
                    for ind_pred in indices_pred: #same as above but for prediction instead of ground truth, notice I use original captions with target and predicted ones with prediction 
                        answer_indicies = []
                        for index, i in enumerate(prediction[ind_pred::]):
                            if i.item() ==0:
                                break
                            offseted_index = index+ ind_pred
                            answer_indicies.append(offseted_index)

                        answers_tokens = input_tokens[ind_pred: offseted_index+1]
                        # for tok in answers_tokens:
                        ans_text = tokenizer.decode(answers_tokens, skip_special_tokens=True)
                        answers_p.append(ans_text)
                          
                    answers_t = " ".join(answers_t) #concat all the answers of one sample for f-score, not sure if there is a better way but I guess it works fine
                    answers_p = " ".join(answers_p)
                    answers_pred.append(answers_p) 

                    f_scores.append(compute_f1(answers_t, answers_p))                
                return  np.array(f_scores), answers_pred




def validation_down(things_dataloader_val, model):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none") 
    with tqdm( unit='it', total=len(things_dataloader_val)) as pbar:
        threshold = 0.5
        exact_scores = []
        running_loss = 0.0
        with torch.no_grad():
            for  it,(label, tokenized_input, attn_mask, answer, caption, ques) in  enumerate(things_dataloader_val):
                label ,tokenized_input, attn_mask = label.to(device), tokenized_input.to(device), attn_mask.to(device)
                outputs = model(tokenized_input, attn_mask)
             


#Masking the question tokens positions in logits for loss calculations
                sep_token = 2
                sep_indices = (tokenized_input == sep_token).int().argmax(dim=1)
                # breakpoint()
                mask = torch.arange(tokenized_input.cpu().size(1)).expand(tokenized_input.cpu().size(0), -1) > sep_indices.cpu().unsqueeze(1)
                # expanded_mask = mask.unsqueeze(-1).expand(-1, -1, outputs.size(2))
                mask = mask.to("cuda")
                # outputs = expanded_mask*outputs
                outputs = outputs.permute(0,2,1) # Permute for Cross Entropy loss
                pred = outputs.argmax(dim=1)
                f1_scores = []
                answers_pred = []
                answers_target = []

                for prediction, target, input_tokens in zip(pred,label, tokenized_input):
                    indices_target = [loc for loc, val in enumerate(target) if val == 1] 
                    indices_pred =  [loc for loc, val in enumerate(prediction) if val == 1] 
                    answers_t = []

                    for ind_target in indices_target:
                        answer_indicies = []
                        for index,i in enumerate(target[ind_target::]):
                            if i.item() ==0:
                                break
                            offseted_index = index + ind_target
                            answer_indicies.append(offseted_index)
                        answers_tokens = input_tokens[ind_target: offseted_index+1]
                        # for tok in answers_tokens:
                        ans_text = tokenizer.decode(answers_tokens, skip_special_tokens=True)
                        answers_t.append(ans_text)
                    answers_target.append(answers_t)
                    answers_p = []
                    
                    for ind_pred in indices_pred:
                        answer_indicies = []
                        for index, i in enumerate(prediction[ind_pred::]):
                            if i.item() ==0:
                                break
                            offseted_index = index+ ind_pred
                            answer_indicies.append(offseted_index)

                        answers_tokens = input_tokens[ind_pred: offseted_index+1]
                        # for tok in answers_tokens:
                        ans_text = tokenizer.decode(answers_tokens, skip_special_tokens=True)
                        answers_p.append(ans_text)
                    answers_pred.append(answers_p)       
                    answers_t = " ".join(answers_t)
                    answers_p = " ".join(answers_p)
                    # print(answers_t)
                    # print(answers_p)
                    print("#############################################################")
                    exact_scores.append(compute_exact(answers_t, answers_p))
                    f1_scores.append(compute_f1(answers_t, answers_p))
                    # f1_macro = f1_score(target.cpu(), prediction.cpu(), average='macro')
                    # f_scores.append(f1_macro)

                print(answers_t)
                print(answers_p)
                print("#############################################################")

                loss = loss_fn(outputs, label)
                loss = loss * mask # Mask out question tokens for loss
                loss = loss.mean()
                # loss = outputs.loss
                this_loss = loss.item()
                running_loss += this_loss
        running_loss = running_loss / len(things_dataloader_val)

        exact_scores = torch.tensor(exact_scores).type(torch.float)
        f1_scores = torch.tensor(f1_scores).type(torch.float)
        exact_score = 100. * exact_scores.mean().item()
        f1_score = 100. * f1_scores.mean().item()
    model.train()
    return f1_score, running_loss
    
def recap_validation(model_captioning, processor, model_qa, val_dataloader):
    '''
    Validation on subset of validation set
    '''
    device = 'cuda:0'

    model_captioning.to(device)
    model_captioning.eval()
    tokenizer = val_dataloader.dataset.tokenizer
    exact_scores = 0.0
    f1_scores = 0.0
    running_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none") 

    # generate captions using updated model
    with tqdm( unit='it', total=len(val_dataloader)) as pbar:
        for  it, (label, answer, captions, ques, images, prompts, image_id) in  enumerate(val_dataloader):
            label, images, prompts = label.to(device), images.to(device), prompts.to(device)
            with torch.no_grad():
                ## Greedy decode captioning model
                generated_tokens_greedy = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=50)
                generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True)
                
                key_indices = range(len(image_id[0]))
                generated_caption_baseline_dict = {key: [value] for key, value in zip(key_indices, generated_caption_baseline)}

                ## key can repeat if same image exists. Hence use index
                caption_dict = {idx: ref_caps_train[key] for idx, key in enumerate(image_id[0])}

                # Downstream Eval
                tokenized = tokenizer(ques, generated_caption_baseline, padding='longest', return_tensors="pt")
                # tokenized = tokenizer(ques, captions, padding='longest', return_tensors="pt")

                tokenized_inputs = tokenized["input_ids"]
                attn_mask = tokenized["attention_mask"]
                outputs = model_qa(tokenized_inputs, attn_mask)
                outputs = outputs.permute(0,2,1) # Permute for Cross Entropy loss
                pred = outputs.argmax(dim=1)
                f_scores = []
                answers_pred = []
                answers_target = []

                for prediction, target, input_tokens, answer_gold in zip(pred,label, tokenized_inputs, answer):
                    indices_pred =  [loc for loc, val in enumerate(prediction) if val == 1] #get all answers beginning of prediction
                    answers_t = []
                    answers_p = []
                    for ind_pred in indices_pred: #same as above but for prediction instead of ground truth, notice I use original captions with target and predicted ones with prediction 
                        answer_indicies = []
                        for index, i in enumerate(prediction[ind_pred::]):
                            if i.item() ==0:
                                break
                            offseted_index = index+ ind_pred
                            answer_indicies.append(offseted_index)

                        answers_tokens = input_tokens[ind_pred: offseted_index+1]
                        # for tok in answers_tokens:
                        ans_text = tokenizer.decode(answers_tokens, skip_special_tokens=True)
                        answers_p.append(ans_text)
                    answer_gold = list(dict.fromkeys(answer_gold))
                    answers_t = " ".join(answer_gold) #concat all the answers of one sample for f-score, not sure if there is a better way but I guess it works fine
                    answers_p = " ".join(answers_p)
                    answers_pred.append(answers_p) 

                    f_scores.append(compute_f1(answers_t, answers_p))   
                    # breakpoint()
                    sep_token = 2
                    sep_indices = (tokenized_inputs == sep_token).int().argmax(dim=1)
                    # breakpoint()
            #         mask = torch.arange(tokenized_inputs.cpu().size(1)).expand(tokenized_inputs.cpu().size(0), -1) > sep_indices.cpu().unsqueeze(1)
            #         # expanded_mask = mask.unsqueeze(-1).expand(-1, -1, outputs.size(2))
            #         mask = mask.to("cuda")
            #         loss = loss_fn(outputs, label)
            #         loss = loss * mask # Mask out question tokens for loss
            #         loss = loss.mean()
            #         # loss = outputs.loss
            #         this_loss = loss.item()
            #         running_loss += this_loss
            # running_loss = running_loss / len(val_dataloader)
            # pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
        print(np.array(f_scores).mean())
        return np.array(f_scores).mean()
    
def compute_score_batch(outputs, label, tokenized_input, answers, tokenizer, tokenized_inputs_gold):
                # tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
                outputs = outputs.permute(0,2,1) # Permute for Cross Entropy loss
                pred = outputs.argmax(dim=1)
                f_scores = []
                answers_pred = []
                answers_target = []

                for prediction, target, input_tokens, gold_tokens, answer_gold in zip(pred,label, tokenized_input, tokenized_inputs_gold, answers):
                    indices_target = [loc for loc, val in enumerate(target) if val == 1] #get all answers begninning of target
                    indices_pred =  [loc for loc, val in enumerate(prediction) if val == 1] #get all answers beginning of prediction
                    answers_t = []

                    # for ind_target in indices_target:
                    #     answer_indicies = []
                    #     for index,i in enumerate(target[ind_target::]): #loop to get all inside answers indecies
                    #         if i.item() ==0:
                    #             break #end of answer
                    #         offseted_index = index + ind_target #add answer beginning offset to get inside index (rest of answer)
                    #         answer_indicies.append(offseted_index) 
                    #     answers_tokens = gold_tokens[ind_target: offseted_index+1] #get answer
                    #     # for tok in answers_tokens:
                    #     ans_text = tokenizer.decode(answers_tokens, skip_special_tokens=True) # decode it
                    #     answers_t.append(ans_text) #all answers of one sample
                    # answers_target.append(answers_t) #Not used, just collecting all answers
                    answers_p = []
                    
                    for ind_pred in indices_pred: #same as above but for prediction instead of ground truth, notice I use original captions with target and predicted ones with prediction 
                        answer_indicies = []
                        for index, i in enumerate(prediction[ind_pred::]):
                            if i.item() ==0:
                                break
                            offseted_index = index+ ind_pred
                            answer_indicies.append(offseted_index)

                        answers_tokens = input_tokens[ind_pred: offseted_index+1]
                        # for tok in answers_tokens:
                        ans_text = tokenizer.decode(answers_tokens, skip_special_tokens=True)
                        answers_p.append(ans_text)
                    answers_pred.append(answers_p)       
                    answer_gold = list(dict.fromkeys(answer_gold))
                    answers_t = " ".join(answer_gold) #concat all the answers of one sample for f-score, not sure if there is a better way but I guess it works fine
                    answers_p = " ".join(answers_p)
                    f_scores.append(compute_f1(answers_t, answers_p))                
                return  np.array(f_scores), answers_pred
import copy
import gc
def is_sublist_numpy(cap, ans):
    cap_array = np.array(cap)
    ans_array = np.array(ans)
    window_size = len(ans_array)
    
    # Check if the ans array is longer than the cap array
    if window_size > len(cap_array):
        return False
    
    # Create a sliding window view of the cap array
    sliding_windows = np.lib.stride_tricks.sliding_window_view(cap_array, window_shape=window_size)
    
    # Check if any of the sliding windows match the ans array
    return any(np.array_equal(window, ans_array) for window in sliding_windows)
    
def check_memory(cuda_device):
    """ Check the total memory and occupied memory for GPU """
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used
def recaptioning(things_coco_dataloader_train, coco_dataloader_val, coco_cider_dataloader, ref_caps, model_captioning, model_downstream, optimizer1, optimizer2, weights):
    # model_downstream_state_dict = torch.load('Downstream_vqa_BIO.pth')
    # model_downstream_pretune = copy.deepcopy(model_downstream)
    # model_downstream_state_dict_base = torch.load('Downstream_vqa_BIO.pth')
    # model_downstream_pretune.module.load_state_dict(model_downstream_state_dict_base)

    # model_downstream_state_dict = torch.load('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/Downstream_vqa_BIO_new.pth')
    # model_downstream_state_dict = torch.load('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/Recaption/Downstream_vqa_BIO_new_linear.pth')


    # model_downstream.load_state_dict(model_downstream_state_dict)
    os.environ["WANDB_API_KEY"] = WANDB_KEY
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    # model_captioning_state_dict = torch.load('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/recaptioning_contrastive_20.pth', map_location='cpu')
    # model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
    # breakpoint()
    model_captioning_state_dict = torch.load("/workspace/T5_pod/recaptioning_pali.pth", map_location='cpu')
    model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
    for name, param in model_captioning.model_blip.named_parameters():
        param.requires_grad = False
    for name, param in model_captioning.model_blip.multi_modal_projector.named_parameters():
        param.requires_grad = True
    # for name, param in model_captioning.model_blip.language_projection.named_parameters():
    #     param.requires_grad = True

    # model_downstream = DDP(model_downstream.module)
    for name, param in model_downstream.named_parameters():
          param.requires_grad = True
    # model_captioning = DDP(model_captioning, device_ids=[0])
    model_captioning = model_captioning.to("cuda:0")
    # model_downstream = model_downstream.to("cuda:0")
    device = "cuda:0"
    # model_base = blip_recaption()
    # model_base = DDP(model_base)
    # model_base = model_base.to("cuda:1")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
    # tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    best_val = 10000
    configs = {"captioning LR": LR,
               "downstream LR": LR_DOWN,
               "batch_size": batch_size
               
               }
    logging_step = 0
    running_avg = 0.0
    running_length = 0.0
    with wandb.init(project="recap", mode=WANDB_STATUS, notes=NOTES, name=EXP_NAME, config= configs):
        for i in range(0,4):
            print("??????????????????????????NEW EPOCH??????????????????????????????????")
            running_loss1 = 0.
            running_loss2 = 0.
            # T = validation_down(coco_dataloader_val, model_downstream)
            inputs_fine = []
            labels_fine = []
            img_ids_fine = []
            captions_fine = []
            attn_mask_fine = []
            labels_gold = []
            # model = AutoModelForCausalLM.from_pretrained("bczhou/TinyLLaVA-3.1B", trust_remote_code=True)
            # model = model.to("cuda:0")

            # c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive = evaluation(model_captioning, coco_dataloader_val, ref_caps)
            # print("##########CIDEr_contrastive########: "+ str(c_contrastive))
            # print("##########CIDEr_beam########: "+ str(c_beam))
            # print("##########rouge_contrastive########: "+ str(r_contrastive))
            # print("##########rouge_beam########: "+ str(r_beam))
            # print("##########recall_contrastive########: "+ str(recall_contrastive))
            # print("###########model##########:" + str("base"))
            # with open("dict_base.jsonl", "w") as f:
            #     for item in dict_base:
            #         f.write(json.dumps(item) + "\n")
#             x=0
#             trained_models = [
#                               "/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/paligemma_coco_all_final20.pth",
                            
# ]
#             for mod in trained_models:
#                     # x+=1
#                     try:
#                         model_captioning_state_dict = torch.load(mod, map_location='cpu')
#                         model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
#                     except:
#                         print("bad model"+str(mod))
#                         continue
#                     del model_captioning_state_dict
#                     c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive, dict_recap = evaluation(model_captioning, coco_dataloader_val, ref_caps)
#                     with open("dict_pali_repetition.jsonl", "w") as f:
#                         for item in dict_recap:
#                             f.write(json.dumps(item) + "\n")
#                     print("##########CIDEr_contrastive########: "+ str(c_contrastive))
#                     print("##########CIDEr_beam########: "+ str(c_beam))
#                     print("##########rouge_contrastive########: "+ str(r_contrastive))
#                     print("##########rouge_beam########: "+ str(r_beam))
#                     print("##########recall_contrastive########: "+ str(recall_contrastive))
#                     print("###########model##########:" + str(mod))

            # c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive, recall_beam, dict_base, dict_base_beam = evaluation(model_captioning, coco_dataloader_val, ref_caps)

            # with open("dict_base_pali_contrastive.jsonl", "w") as f:
            #     for item in dict_base:
            #         f.write(json.dumps(item) + "\n")
            # with open("dict_base_pali_beam.jsonl", "w") as f:
            #     for item in dict_base_beam:
            #         f.write(json.dumps(item) + "\n")
            # print("##########CIDEr_contrastive########: "+ str(c_contrastive))
            # print("##########CIDEr_beam########: "+ str(c_beam))
            # print("##########rouge_contrastive########: "+ str(r_contrastive))
            # print("##########rouge_beam########: "+ str(r_beam))
            # print("##########recall_contrastive########: "+ str(recall_contrastive))
            # print("##########recall_beam########: "+ str(recall_beam))
            # print("###########model##########:" "base")

#             trained_models = [
                                
#                               "/l/users/abdelrahman.mohamed/pali/checkpoints/pali_5e-7_eos_055.pth",

                            
# ]
#             for mod in trained_models:
#                     try:
#                         model_captioning_state_dict = torch.load(mod, map_location='cpu')
#                         model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
#                     except:
#                         print("bad model"+str(mod))
#                     del model_captioning_state_dict
#                     c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive, recall_beam, dict_base, dict_base_beam = evaluation(model_captioning, coco_dataloader_val, ref_caps)

#                     with open("dict_0_55_pali_contrastive.jsonl", "w") as f:
#                         for item in dict_base:
#                             f.write(json.dumps(item) + "\n")
#                     with open("dict_0_55_pali_beam.jsonl", "w") as f:
#                         for item in dict_base_beam:
#                             f.write(json.dumps(item) + "\n")
#                     print("##########CIDEr_contrastive########: "+ str(c_contrastive))
#                     print("##########CIDEr_beam########: "+ str(c_beam))
#                     print("##########rouge_contrastive########: "+ str(r_contrastive))
#                     print("##########rouge_beam########: "+ str(r_beam))
#                     print("##########recall_contrastive########: "+ str(recall_contrastive))
#                     print("##########recall_beam########: "+ str(recall_beam))
#                     print("###########model##########:" + str(mod))



#             trained_models = [
                                
#                               "/l/users/abdelrahman.mohamed/pali/checkpoints/pali_5e-7_eos_065.pth",

                            
# ]
#             for mod in trained_models:
#                     try:
#                         model_captioning_state_dict = torch.load(mod, map_location='cpu')
#                         model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
#                     except:
#                         print("bad model"+str(mod))
#                     del model_captioning_state_dict
#                     c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive, recall_beam, dict_base, dict_base_beam = evaluation(model_captioning, coco_dataloader_val, ref_caps)

#                     with open("dict_0_65_pali_contrastive.jsonl", "w") as f:
#                         for item in dict_base:
#                             f.write(json.dumps(item) + "\n")
#                     with open("dict_0_65_pali_beam.jsonl", "w") as f:
#                         for item in dict_base_beam:
#                             f.write(json.dumps(item) + "\n")
#                     print("##########CIDEr_contrastive########: "+ str(c_contrastive))
#                     print("##########CIDEr_beam########: "+ str(c_beam))
#                     print("##########rouge_contrastive########: "+ str(r_contrastive))
#                     print("##########rouge_beam########: "+ str(r_beam))
#                     print("##########recall_contrastive########: "+ str(recall_contrastive))
#                     print("##########recall_beam########: "+ str(recall_beam))
#                     print("###########model##########:" + str(mod))

#             trained_models = [
                                
#                               "/l/users/abdelrahman.mohamed/pali/checkpoints/pali_norepeat_last3_eos_025.pth",

                            
# ]
#             for mod in trained_models:
#                     try:
#                         model_captioning_state_dict = torch.load(mod, map_location='cpu')
#                         model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
#                     except:
#                         print("bad model"+str(mod))
#                     del model_captioning_state_dict
#                     c_contrastive, c_beam, r_contrastive, r_beam, recall_contrastive, recall_beam, dict_base, dict_base_beam = evaluation(model_captioning, coco_dataloader_val, ref_caps)

#                     with open("dict_0_7_pali_contrastive.jsonl", "w") as f:
#                         for item in dict_base:
#                             f.write(json.dumps(item) + "\n")
#                     with open("dict_0_7_pali_beam.jsonl", "w") as f:
#                         for item in dict_base_beam:
#                             f.write(json.dumps(item) + "\n")
#                     print("##########CIDEr_contrastive########: "+ str(c_contrastive))
#                     print("##########CIDEr_beam########: "+ str(c_beam))
#                     print("##########rouge_contrastive########: "+ str(r_contrastive))
#                     print("##########rouge_beam########: "+ str(r_beam))
#                     print("##########recall_contrastive########: "+ str(recall_contrastive))
#                     print("##########recall_beam########: "+ str(recall_beam))
#                     print("###########model##########:" + str(mod))

#             quit()

#             # s = evalu
#             # s = evaluation(model_captioning, things_coco_dataloader_train, ref_caps)

#             # model_downstream.eval()
            # breakpoint()
            # print("##########CIDEr########: "+ str(s))
            # print("##########F1-Score########: "+ str(f_score))


            with tqdm( unit='it', total=len(things_coco_dataloader_train)) as pbar:



                    for  it,(label, answer, captions, ques, images, prompts, image_id) in  enumerate(things_coco_dataloader_train):
                        # if it%250==0:
                        #     vqa_fine_rec = Coco_Dataset_VQA(split = "finetune", downstream= False)
                            
                        #     vqa_fine_dataloader_rec = DataLoader(vqa_fine_rec, batch_size=32, shuffle=True, num_workers=1,collate_fn = vqa_fine_rec.collate_fn,
                        #                 drop_last=True)
                        #     finetune_down(vqa_fine_dataloader_rec, model_captioning, model_downstream, optimizer1, optimizer2, None)
                        #     del vqa_fine_rec
                        #     del vqa_fine_dataloader_rec
                        #     gc.collect()
                        if it%75==0:
                            one_gram = 0.0
                            bi_gram = 0.0
                            tri_gram = 0.0
                            quad_gram = 0.0
                            captions_greedy, captions_beam, probs_mean, rest_probs_mean, avg_length, recalls, n_grams= val_evaluation(model_captioning, coco_dataloader_val, ref_caps)
                            wandb.log({"eos_val_prob":probs_mean})
                            wandb.log({"rest_val_prob":rest_probs_mean})
                            wandb.log({"val_length":avg_length})
                            wandb.log({"val_recall":recalls})
                            wandb.log({"avg_repetiton":n_grams})


                            for i in captions_greedy.values():
                                one_gram +=  distinct_n_sentence_level(i[0],1)
                            for i in captions_greedy.values():
                                bi_gram +=  distinct_n_sentence_level(i[0],2)
                            for i in captions_greedy.values():
                                tri_gram +=  distinct_n_sentence_level(i[0],3)
                            for i in captions_greedy.values():
                                quad_gram +=  distinct_n_sentence_level(i[0],4)
                            model_captioning.train()
                            print(captions_greedy)
                            print(captions_beam)
                            print(probs_mean)
                            print(n_grams)
                            
                            wandb.log({"n1":one_gram/len(captions_greedy)})
                            wandb.log({"n2":bi_gram/len(captions_greedy)})
                            wandb.log({"n3":tri_gram/len(captions_greedy)})
                            wandb.log({"m4":quad_gram/len(captions_greedy)})
                            # if probs_mean<0.7 and probs_mean >0.6:
                            #     torch.save(model_captioning.state_dict(), './recaptioning_T5_con_last4_eos_065.pth')
                            # if probs_mean<0.6 and probs_mean >0.5:
                            #     torch.save(model_captioning.state_dict(), './recaptioning_T5_con_last4_eos_055.pth')
                            # if probs_mean<0.5 and probs_mean >0.4:
                            #     torch.save(model_captioning.state_dict(), './recaptioning_T5_con_last4_eos_045.pth')
                            # if probs_mean<0.4 and probs_mean >0.3:
                            #     torch.save(model_captioning.state_dict(), './recaptioning_T5_con_last4_eos_035.pth')
                            # if probs_mean<0.3 and probs_mean >0.2:
                            #     torch.save(model_captioning.state_dict(), './recaptioning_T5_con_last4_eos_025.pth')
                            if probs_mean<0.3 and probs_mean >0.2:
                                    quit()
                    # with torch.autograd.set_detect_anomaly(True):
                        label, images, prompts = label.to(device), images.to(device), prompts.to(device)
                        # images_base, prompts_base = images.to("cuda:1"), prompts.to("cuda:1")
                        
                        with torch.no_grad():
                            ## Greedy decode captioning model
                            
                            generated_tokens_greedy = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images, max_new_tokens=50)[:,prompts.shape[1]:]
                            generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True)
                            ## Greedy decode base captioning model
                            # generated_tokens_greedy_base = model_base.module.model_blip.generate(input_ids=prompts_base, pixel_values=images_base, max_length=50)
                            # generated_caption_base = processor.batch_decode(generated_tokens_greedy_base, skip_special_tokens=True)
                            # z = [len(x.split(' ')) for x in generated_caption_baseline]
                            generated_caption_baseline_dict = {key: [value] for key,value in zip(image_id[0], generated_caption_baseline)}
                            # generated_caption_dict_base = {key: [value] for key,value in zip(image_id[0], generated_caption_base)}

                        # caption_dict = {key: [value] for key,value in zip(image_id[0], gt_caption)}
                            caption_dict = {key: ref_caps_train[key] for key in image_id[0] } #all refrences for the caption (all the five captions)
                            #  
                            # base_score_r, recall_base = evaluate_rouge(generated_caption_dict_base, caption_dict)
                            recap_score_r, recall_recap = evaluate_rouge(generated_caption_baseline_dict, caption_dict) #use ref caps since we are using dict
                            # base_score_c = evaluate_cider(generated_caption_dict_base, caption_dict)
                            recap_score_c = evaluate_cider(generated_caption_baseline_dict, caption_dict) #use ref caps since we are using dict
                        
                        # total,used = check_memory(2)
                        # total = int(total)
                        # used = int(used)
                        # max_mem = int(total * 0.90)
                        # print('Total memory: ' + str(total) + ', used memory: ' + str(used))
                        # mem = max_mem - used

                        # try:
                        # if it<50 or it>120:
                        # outs  = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images,penalty_alpha=0.6, top_k=2,  max_new_tokens=40, output_logits= True, return_dict_in_generate=True)
                        outs  = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images,  max_new_tokens=40, output_logits= True, return_dict_in_generate=True)
                            
                        # outs = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=40,num_beams= 3, output_logits= True, return_dict_in_generate=True )
                        # outs  = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images, do_sample=True, top_p=0.92, top_k=0 , max_length=35,output_logits= True, return_dict_in_generate=True  )
                        # outs  = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=40,output_logits= True, return_dict_in_generate=True  )

                        # else:
                        # outs  = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images, do_sample=True ,max_length=35,output_logits= True, return_dict_in_generate=True  )
                        
                            # flag = 0
                        # except Exception as f:
                        #     print(f)
                        #     flag = 1
                        #     continue
                        generated_tokens_sampling = outs[0][:,prompts.shape[1]:]
                        logits = outs[1]
                        #what if i did not detach?
                        # reward_img, reward_context = outs[1]
                        # rewards_img, rewards_context = reward_img.detach(), reward_context.detach()
                        # rewards_img = reward_img
                        # del reward_img
                        # del reward_context
                        # rewards = outs[1]
                        logits = torch.stack(logits, dim=1)
                        indices = generated_tokens_sampling.unsqueeze(-1)
                        # words_prob = torch.gather(logits, dim=2, index=indices).squeeze(-1)

######################## using log_probs############################################
                        logits_prob = torch.nn.functional.log_softmax(logits, dim=-1)
                        words_prob = torch.gather(logits_prob, dim=2, index=indices).squeeze(-1)
                        # # print(generated_tokens_sampling)
                        generated_caption = processor.batch_decode(generated_tokens_sampling, skip_special_tokens=True) 
                        
                        generated_caption_random_dict = {key: [value] for key,value in zip(image_id[0], generated_caption)}
                        generated_caption_random_dict_base = {key: [value] for key,value in zip(image_id[0], generated_caption)}

                        # tokenized = tokenizer(ques, captions, padding='longest', return_tensors="pt")
                        # tokenized_inputs_gold = tokenized["input_ids"]
                        # attn_mask_gold = tokenized["attention_mask"]





                        # tokenized = tokenizer(ques, generated_caption, padding='longest', return_tensors="pt")
                        # tokenized_inputs = tokenized["input_ids"]
                        # attn_mask = tokenized["attention_mask"]
                        # outputs = model_downstream(tokenized_inputs.to('cuda:0'), attn_mask.to('cuda:0'))

                        # f1_score_r, answers_recap = compute_score_batch(outputs, label, tokenized_inputs, answer, tokenizer, tokenized_inputs_gold)
                        # # outputs_pretune = model_downstream_pretune(tokenized_inputs, attn_mask)
                        
                        
                        #  
                        # loss_r = loss_fn(outputs, labels) #need accuracy
                        # loss_r = outputs.loss


######################################################################
                        
                        # for token_cap,ansers, id, cap_fine, att, lab_gold in zip(tokenized_inputs, answer, image_id[0], generated_caption, attn_mask, label):
                        #         # model_downstream.train()
                                        
                        #         label_fine = [0] * len(token_cap)
                        #         for ans in ansers:
                        #             token_answer =  tokenizer(" " + ans.lower())['input_ids'][1:-1] #skip beginning of sequence and end of sequence tokens
    
                        #             sep_token_id = tokenizer.sep_token_id
                        #             sep_token_position = list(token_cap).index(sep_token_id) #question end
                        #             # breakpoint()
                        #             if is_sublist_numpy(token_cap[sep_token_position:], token_answer):
                        #                 flag = 0
                        #                 tok = token_answer[0]
                        #                 indices = [loc for loc, val in enumerate(token_cap[sep_token_position::]) if val == tok] #to avoid taking the answer in question if it exists
                        #                 #indecies of correct answers
                                            
                        #                 for index in indices: #loop over possible matches of start token
                        #                     index = sep_token_position + index #add question offset
                        #                     if token_cap[index: index+len(token_answer)].tolist() == token_answer : #check if the answer starting from this start token matches the original answer
                        #                         label_fine[index] = 1 #the start is B
                        #                         if len(token_answer)>1:
                        #                             label_fine[index+1: index+len(token_answer)] = [2]* (len(token_answer)-1) #the rest of the answer tokens are I
                        #         if sum(label_fine) !=0: #answer exists in generated caption
                        #             labels_fine.append(label_fine)
                        #             inputs_fine.append(token_cap.tolist())
                        #             img_ids_fine.append(id)
                        #             captions_fine.append(cap_fine)
                        #             attn_mask_fine.append(att.tolist())
                        #             labels_gold.append(lab_gold.tolist())


                        

                        # Downstream performance with greedy decoded caption
                        # with torch.no_grad():  
                        #     ####### gold captions
      
                        #     outputs_gold = model_downstream( tokenized_inputs_gold.to(device), attn_mask_gold.to(device))
                        #     # breakpoint()
                        #     f1_score_gold, _ = compute_score_batch(outputs_gold, label, tokenized_inputs_gold, answer, tokenizer, tokenized_inputs_gold)
                            
                        #     ######## baseline
                        #     tokenized = tokenizer(ques, generated_caption_baseline, padding='longest', return_tensors="pt")
                        #     tokenized_inputs = tokenized["input_ids"]
                        #     attn_mask = tokenized["attention_mask"]
                        #     outputs_baseline = model_downstream(tokenized_inputs.to(device), attn_mask.to(device))
                        #     f1_score_baseline, _ = compute_score_batch(outputs_baseline, label, tokenized_inputs, answer, tokenizer, tokenized_inputs_gold)
                        #     # loss_baseline = outputs_baseline.loss

                            ####### base model
                            # tokenized = tokenizer(ques, generated_caption_base, padding='longest', return_tensors="pt")
                            # tokenized_inputs = tokenized["input_ids"]
                            # attn_mask = tokenized["attention_mask"]
                            # outputs_base = model_downstream(tokenized_inputs, attn_mask)
                            # f1_score_base = compute_score_batch(outputs_base, label, tokenized_inputs, answer, tokenizer, tokenized_inputs_gold)
              
                        # f1_score_pretune, _ = compute_score_batch(outputs_pretune, label, tokenized_inputs, answer, tokenizer, tokenized_inputs_gold)

#################################################################################
                            # perplexity = evaluate.load("perplexity", module_type="metric")
                            # # breakpoint()
                            # prep_recaption = perplexity.compute(model_id='facebook/opt-2.7b',add_start_token=False,predictions=generated_caption_baseline, device="cuda")
                            # prep_base = perplexity.compute(model_id='facebook/opt-2.7b',add_start_token=False,predictions=generated_caption_base, device="cuda")
                            # prep = torch.tensor(prep_recaption["perplexities"]) - torch.tensor(prep_base["perplexities"])
################################################################################
                        # if args.no_baseline:
                        #     f1_score_baseline = 0.5
                        # reward = f1_score_r - f1_score_baseline.mean()
                        # breakpoint()
                        # reward = scale_rewards(reward)
                        # reward_anchor_r = recall_recap - recall_base
                        # reward_anchor_c = recap_score_c - base_score_c
                        if recall_recap.mean() < 0.1:
                            break

                        ### The function return distincit n-gram/ len sentence, so it ranges from 0 to 1 according to how many distinct n-gram (in case all distnct=1)
                        # distinct = [] 
                        # for recap, base in zip(generated_caption_baseline, generated_caption_base):
                        #     distinct_base =  distinct_n_sentence_level(base,1) + distinct_n_sentence_level(generated_caption_base,2) + distinct_n_sentence_level(generated_caption_base,3) 
                        #     distinct_recap =  distinct_n_sentence_level(recap,1) + distinct_n_sentence_level(generated_caption_baseline,2) + distinct_n_sentence_level(generated_caption_base,3) 
                        #     distinct.append(distinct_recap - distinct_base)
                        # print(distinct_recap)
                        # print(distinct_base)
                        #   #check here
                        #  
                        # try:distinct
                        min_val = -0.6
                        max_val = 0.15
                        scale = 1.0

                        # Generate 10 left-skewed random numbers
                        # exp_random = np.random.exponential(scale)  # Generate exponential random value
                        # skewed_value = max_val - (exp_random / (exp_random + 1)) * (max_val - min_val)  # Calculate the left-skewed value
                        # reward = skewed_value
                        # total_reward = -0.2
                        # total_reward = reward

                        # except:
                        #     pad = reward.size(0)-reward_anchor_r.shape[0] # for some reason some of the batches have smaller rouge vector
                        #     reward_anchor_r = np.pad(reward_anchor_r,pad,'minimum')[pad::] #it pads both sides, ignore the left padding
                        #     total_reward = reward + reward_anchor_r
                        #     print("smaller rouge vector")
                        # print(distinct)
                        # print(reward_anchor_c)
                        # loss = -torch.mean(words_prob, -1) * torch.tensor(rewards).cuda()
                        # rewards = rewards_img + 0.5* rewards_context
                        # rewards = rewards_img
                        #Mask newline Token
                        # for i in range (generated_tokens_sampling.cpu().shape[0]):
                        #     try:
                        #         idx = np.where(generated_tokens_sampling.cpu()[i] == 50118)[0][0]
                        #         rewards[i, idx:] = 0 
                        #     except:
                        #         continue
                        rewards = []
                        # rewards = reward_img
                        # breakpoint()
                        for i in range (generated_tokens_sampling.cpu().shape[0]):
                            tensor = []
                            for k in range (generated_tokens_sampling.cpu().shape[1]):
                                if generated_tokens_sampling[i,k].item() == 1:
                                    tensor.append(-1)
                                else: 
                                    tensor.append(0)
                            rewards.append(tensor)
                        rewards= torch.tensor(rewards).to("cuda:0")
                        loss = -torch.mean(words_prob * rewards, -1)
                        # loss = -torch.mean(rewards,-1)*15

                        # logits_prob = logits_prob.mean()
                        # loss = -torch.mean(logits_prob, -1) * torch.tensor(total_reward).cuda()

                        #  logits_prob
                        loss = loss.mean()
                        loss.backward()
                        # optimizer1.step()
                        # optimizer1.zero_grad()
                        


                        if it % 3 == 0:
                            optimizer1.step()
                            optimizer1.zero_grad()
                        

                            # model_downstream.train()

                            # max_length = max(len(sublist) for sublist in labels_fine)

                            # # Pad the lists with a default value (e.g., 0)
                            # labels_fine = [sublist + [0] * (max_length - len(sublist)) for sublist in labels_fine]
                            # max_length = max(len(sublist) for sublist in inputs_fine)

                            # # Pad the lists with a default value (e.g., 0)
                            # inputs_fine = [sublist + [1] * (max_length - len(sublist)) for sublist in inputs_fine]
                            # # Pad the lists with a default value (e.g., 0)
                            # max_length = max(len(sublist) for sublist in attn_mask_fine)

                            # # Pad the lists with a default value (e.g., 0)
                            # attn_mask_fine = [sublist + [0] * (max_length - len(sublist)) for sublist in attn_mask_fine]

                            # max_length = max(len(sublist) for sublist in labels_gold)

                            # # Pad the lists with a default value (e.g., 0)
                            # labels_gold = [sublist + [0] * (max_length - len(sublist)) for sublist in labels_gold]




                            # labels_fine = torch.tensor(labels_fine).to(device)
                            # inputs_fine = torch.tensor(inputs_fine).to(device)
                            # attn_mask_fine = torch.tensor(attn_mask_fine).to(device)
                            # labels_gold = torch.tensor(labels_gold).to(device)
                            # outputs_fine = model_downstream(inputs_fine, attn_mask_fine)
                            # sep_token = 2
                            # sep_indices = (inputs_fine == sep_token).int().argmax(dim=1)
                            # # breakpoint()
                            # mask = torch.arange(inputs_fine.cpu().size(1)).expand(inputs_fine.cpu().size(0), -1) > sep_indices.cpu().unsqueeze(1)
                            # # expanded_mask = mask.unsqueeze(-1).expand(-1, -1, outputs.size(2))
                            # mask = mask.to("cuda")
                            # # outputs = expanded_mask*outputs
                            # outputs_fine = outputs_fine.permute(0,2,1) # Permute for Cross Entropy loss
                            # loss = loss_fn(outputs_fine, labels_fine)
                            # loss = loss * mask # Mask out question tokens for loss
                            # loss = loss.mean()
                            # # loss = outputs.loss
                            # this_loss = loss.item()
                            # #loss = loss_fn(outputs, labels.float())
                            # loss.backward()
                            # optimizer2.step()
                            # optimizer2.zero_grad()
                            # gc.collect()
                            # inputs_fine = []
                            # labels_fine = []
                            # img_ids_fine = []
                            # captions_fine = []
                            # attn_mask_fine = []
                            # labels_gold = []
                            # torch.cuda.empty_cache()

                            # model_downstream.eval()

                        # optimizer1.step()
                        this_loss = loss.item()
                        running_loss1 += this_loss


                        # if it%2==0:
                        #     #  
                        #     # print(distinct_base)
                        #     # print(distinct_recap)
                        #     print("#####################################it#######################" )
                        #     print(it)
                        #     # print(recap_score_c)
                        #     # print("####################ques#######################")
                        #     # ques = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), ques)}
                        #     # print(ques)
                        #     # print("####################base#######################")
                        #     # print(generated_caption_dict_base)  
                        #     print("####################GT#######################")
                        #     caption_dict = {key: [value] for key,value in zip(image_id[0], captions)}
                        #     print(caption_dict)
                        #     print("###################recap########################")
                        #     print(generated_caption_baseline_dict)
                        #     # print("####################answer#####################")
                        #     # answers = [list(set(ans)) for ans in answer]
                        #     # ans = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), answers)}
                        #     # print(ans)
                        #     # print("####################answer gold#####################")
                        #     # ans_gold = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), answers_gold)}
                        #     # print(ans_gold)
                        #     # print("####################answer_recap#####################")
                        #     # answers_recap = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), answers_recap)}
                        #     # print(answers_recap)
                        #     # print("####################scores#####################")
                        #     # scores = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), f1_score_r)}
                        #     # print(scores)
                        #     # print("####################scores GT#####################")
                        #     # # scores_gt = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), f1_score_gold)}
                        #     # print(scores_gt)
                        #     print("###################sampling########################")
                        #     print(generated_caption_random_dict_base)

                            # print(loss_r)
                        
                            # s = evaluation(model_captioning, coco_cider_dataloader, ref_caps)
                        if it%300==0:
                                torch.save(model_captioning.state_dict(), './recaptioning_pali.pth')
                        pbar.set_postfix(loss=this_loss)
                        pbar.update()
                        seq_recap = [len(x.split(' ')) for x in generated_caption_baseline]
                        # seq_base = [len(x.split(' ')) for x in generated_caption_base]

                        avg_seq_recap = np.mean(seq_recap)
                        running_length +=avg_seq_recap
                        running_avg = running_length/it
                        # if running_avg>20:
                        #     try:
                        #         torch.save(model_captioning.state_dict(), './paligemma_coco_all_final'+str(int(running_avg))+'.pth')
                        #     except:
                        #         print("failed to save")
                            
                        
                        # avg_seq_base = np.mean(seq_base)
                        # net_avg_seq = avg_seq_recap - avg_seq_base
                        logging_step += 1
                        wandb.log({"recall": recall_recap.mean()})
                        # wandb.log({"n-gram penality": np.array(distinct).mean()})
                        # wandb.log({"f-score-base": f1_score_base.mean()})
                        # wandb.log({"f-score-finetune": f1_score_r.mean()})
                        # wandb.log({"f-score-pretune": f1_score_pretune.mean()})

                        # wandb.log({"f-score-gold": f1_score_gold.mean()})
                        wandb.log({"cider_recap": recap_score_c})
                        # wandb.log({"text_sim": rewards_context.mean()})
                        wandb.log({"rewards": rewards.float().mean()})

                        # wandb.log({"cider_base": base_score_c})
                        # wandb.log({"reward": rewards.mean()})
                        wandb.log({"loss": this_loss})
                        wandb.log({"average length": avg_seq_recap})
                        # wandb.log({"net average length": net_avg_seq})
                        wandb.log({"epoch": i})
                        wandb.log({"step": logging_step})
                    
            # torch.save(model_captioning.module.state_dict(), './recaptioning_contrastive_1e-7 last'+str(i)+'.pth')
                        # wandb.log({"memory": mem})

                        # wandb.log({"distinct": np.array(distinct_recap).mean()})




def pretraining_down(things_dataloader_train,things_dataloader_val, model, optimizer, weights, scheduler):
    # weights = weights.to(device)
    loss_fn = torch.nn.CrossEntropyLoss( reduction="none") # No reduction, to mask out question tokens
    model_downstream_state_dict = torch.load('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/Downstream_vqa_BIO_new.pth')

    model.module.load_state_dict(model_downstream_state_dict)

    f1_score, val_loss = validation_down(things_dataloader_val, model)
    print(f1_score)
    for name, param in model.module.model.named_parameters():
        param.requires_grad = True
    # breakpoint()
    # for name, param in model.module.model[1].named_parameters():
    #     param.requires_grad = True
    # loss_fn = MultiClassBCELoss()
    # model_downstream_state_dict = torch.load('Downstream_vqa_really_what_only_augmented_gleu.pth')
    # model.module.load_state_dict(model_downstream_state_dict)
    # model = DDP(model.module)
    # model = model.to(device)
    best_val = 10000

    for i in range(0,1000):
        running_loss = 0.
        with tqdm( unit='it', total=len(things_dataloader_train)) as pbar:
            for  it,(label, tokenized_input, attn_mask, answer, caption, ques) in  enumerate(things_dataloader_train):
                label ,tokenized_input, attn_mask = label.to(device), tokenized_input.to(device), attn_mask.to(device)
                outputs = model(tokenized_input, attn_mask)
             


#Masking the question tokens positions in logits for loss calculations
                sep_token = 2
                sep_indices = (tokenized_input == sep_token).int().argmax(dim=1)
                # breakpoint()
                mask = torch.arange(tokenized_input.cpu().size(1)).expand(tokenized_input.cpu().size(0), -1) > sep_indices.cpu().unsqueeze(1)
                # expanded_mask = mask.unsqueeze(-1).expand(-1, -1, outputs.size(2))
                mask = mask.to("cuda")
                # outputs = expanded_mask*outputs
                outputs = outputs.permute(0,2,1) # Permute for Cross Entropy loss
                
                # breakpoint()
                
                # breakpoint()

                    # breakpoint()

                loss = loss_fn(outputs, label)
                loss = loss * mask # Mask out question tokens for loss
                loss = loss.mean()
                # loss = outputs.loss
                this_loss = loss.item()
                running_loss += this_loss
                #loss = loss_fn(outputs, labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

                loss = running_loss / len(things_dataloader_train)
            f1_score, val_loss = validation_down(things_dataloader_val, model)

        scheduler.step(val_loss)

        if val_loss < best_val:
         patience = 0
         best_val = val_loss
         torch.save(model.module.state_dict(), 'Downstream_vqa_BIO_new.pth')
        else:
            patience+=1
        if patience ==50:
            break
        print("##########VAL LOSS########: "+ str(val_loss))
        print("##########accuracy########: "+ str(np.mean(f1_score)))
        # print("##########accuracy2########: "+ str(acc2))


import copy

def finetune_down(things_coco_dataloader_val, model_captioning, model_downstream, optimizer1, optimizer2, weights):
    best_f = 0
    # os.environ["WANDB_API_KEY"] = WANDB_KEY
    loss_fn = torch.nn.CrossEntropyLoss( reduction="none") # No reduction, to mask out question tokens
    # for name, param in model_captioning.model_blip.named_parameters():
    #     param.requires_grad = False
    # for name, param in model_captioning.model_blip.qformer.named_parameters():
    #     param.requires_grad = False
    # for name, param in model_captioning.model_blip.language_projection.named_parameters():
    #     param.requires_grad = False
    # model_downstream = DDP(model_downstream.module)
    # model_downstream = model_downstream.to("cuda")
    for name, param in model_downstream.named_parameters():
          param.requires_grad = True
    # model_captioning = DDP(model_captioning, device_ids=[0])
    # model_captioning = model_captioning.to("cuda:0")
    device = "cuda:0"
    # model_base = blip_recaption(dtype_precision=DTYPE_PRECISION) #dtype_precision=DTYPE_PRECISION
    # model_base = DDP(model_base)
    # model_base = model_base.to("cuda:1")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model_downstream.train()
    best_val = 10000
    configs = {"captioning LR": LR,
               "downstream LR": LR_DOWN,
               
               }
    logging_step = 0
    inputs_fine = []
    labels_fine = []
    img_ids_fine = []
    captions_fine = []
    attn_mask_fine = []
    labels_gold = []
    valid_ans = []
    # with wandb.init(project="recap", mode=WANDB_STATUS, notes=NOTES, name=EXP_NAME, config= configs):
    for i in range(0,1):
            running_loss = 0.
            running_loss2 = 0.
            # T = validation_down(coco_dataloader_val, model_downstream)

            # s,s2 = evaluation(model_captioning, coco_cider_dataloader, ref_caps)
            #  

            # print("##########CIDEr########: "+ str(s))
            # print("##########F1-Score########: "+ str(f_score))
            with tqdm( unit='it', total=len(things_coco_dataloader_val)) as pbar:
                
                for  it,(label, answer, captions, ques, images, prompts, image_id) in  enumerate(things_coco_dataloader_val):
                        
                    # with torch.autograd.set_detect_anomaly(True):
                        label, images, prompts = label.to(device), images.to(device), prompts.to(device)
                        # images_base, prompts_base = images.to("cuda:1", DTYPE_PRECISION), prompts.to("cuda:1")
                        
                        with torch.no_grad():
                            ## Greedy decode captioning model
                        #     generated_tokens_greedy = model_captioning.module.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=50)
                        #     generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True)
                        #     ## Greedy decode base captioning model
                        #     # generated_tokens_greedy_base = model_base.module.model_blip.generate(input_ids=prompts_base, pixel_values=images_base, max_length=50)
                        #     # generated_caption_base = processor.batch_decode(generated_tokens_greedy_base, skip_special_tokens=True)
                        #     # z = [len(x.split(' ')) for x in generated_caption_baseline]
                        #     generated_caption_baseline_dict = {key: [value] for key,value in zip(image_id[0], generated_caption_baseline)}
                        #     # generated_caption_dict_base = {key: [value] for key,value in zip(image_id[0], generated_caption_base)}

                        # # caption_dict = {key: [value] for key,value in zip(image_id[0], captions)}
                        #     caption_dict = {key: ref_caps_train[key] for key in image_id[0] } #all refrences for the caption (all the five captions)
                        #     #  
                        #     # base_score_r, recall_base = evaluate_rouge(generated_caption_dict_base, caption_dict)
                        #     recap_score_r, recall_recap = evaluate_rouge(generated_caption_baseline_dict, caption_dict) #use ref caps since we are using dict
                        #     # base_score_c = evaluate_cider(generated_caption_dict_base, caption_dict)
                        #     recap_score_c = evaluate_cider(generated_caption_baseline_dict, caption_dict) #use ref caps since we are using dict
                             

                            # outs  = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images,penalty_alpha=0.6, top_k=5,  max_length=50 )
                            outs  = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images,do_sample=True,  max_length=60 )

                            
                            #outs  = model_captioning.module.model_blip.generate(input_ids=prompts, pixel_values=images, do_sample=True, max_length=50,output_logits= True, return_dict_in_generate=True  )
    
                        generated_tokens_sampling = outs
                        generated_caption = processor.batch_decode(generated_tokens_sampling, skip_special_tokens=True) 
                        generated_caption_random_dict = {key: [value] for key,value in zip(image_id[0], generated_caption)}
                        # generated_caption_random_dict_base = {key: [value] for key,value in zip(image_id[0], generated_caption)}

                        #     # # 

                        tokenized = tokenizer(ques, generated_caption, padding='longest', return_tensors="pt")
                        tokenized_inputs = tokenized["input_ids"]
                        attn_mask = tokenized["attention_mask"]

                        for token_cap,ansers, id, cap_fine, att, lab_gold in zip(tokenized_inputs, answer, image_id[0], generated_caption, attn_mask, label):
                                label_fine = [0] * len(token_cap)
                                for ans in ansers:
                                    token_answer =  tokenizer(" " + ans.lower())['input_ids'][1:-1] #skip beginning of sequence and end of sequence tokens
    
                                    sep_token_id = tokenizer.sep_token_id
                                    sep_token_position = list(token_cap).index(sep_token_id) #question end
                                    # breakpoint()
                                    if is_sublist_numpy(token_cap[sep_token_position:], token_answer):
                                        # breakpoint()
                                        
                                        flag = 0
                                        tok = token_answer[0]
                                        indices = [loc for loc, val in enumerate(token_cap[sep_token_position::]) if val == tok] #to avoid taking the answer in question if it exists
                                        #indecies of corCoco_Dataset_VQArect answers
                                            
                                        for index in indices: #loop over possible matches of start token
                                            index = sep_token_position + index #add question offset
                                            if token_cap[index: index+len(token_answer)].tolist() == token_answer : #check if the answer starting from this start token matches the original answer
                                                label_fine[index] = 1 #the start is B
                                                if len(token_answer)>1:
                                                    label_fine[index+1: index+len(token_answer)] = [2]* (len(token_answer)-1) #the rest of the answer tokens are I
                                if sum(label_fine) !=0: #answer exists in generated caption
                                    labels_fine.append(label_fine)
                                    inputs_fine.append(token_cap.tolist())
                                    img_ids_fine.append(id)
                                    captions_fine.append(cap_fine)
                                    attn_mask_fine.append(att.tolist())
                                    labels_gold.append(lab_gold.tolist())
                                    valid_ans.append(ansers)
                        if it%5==0:
                            # print("############################captions###################")
                            # print(generated_caption_random_dict)
                            max_length = max(len(sublist) for sublist in labels_fine)

                            # Pad the lists with a default value (e.g., 0)
                            labels_fine = [sublist + [0] * (max_length - len(sublist)) for sublist in labels_fine]
                            max_length = max(len(sublist) for sublist in inputs_fine)

                            # Pad the lists with a default value (e.g., 0)
                            inputs_fine = [sublist + [1] * (max_length - len(sublist)) for sublist in inputs_fine]
                            # Pad the lists with a default value (e.g., 0)
                            max_length = max(len(sublist) for sublist in attn_mask_fine)

                            # Pad the lists with a default value (e.g., 0)
                            attn_mask_fine = [sublist + [0] * (max_length - len(sublist)) for sublist in attn_mask_fine]

                            max_length = max(len(sublist) for sublist in labels_gold)

                            # Pad the lists with a default value (e.g., 0)
                            labels_gold = [sublist + [0] * (max_length - len(sublist)) for sublist in labels_gold]
                            labels_fine = torch.tensor(labels_fine).to(device)
                            inputs_fine = torch.tensor(inputs_fine).to(device)
                            attn_mask_fine = torch.tensor(attn_mask_fine).to(device)
                            labels_gold = torch.tensor(labels_gold).to(device)
                            

                            with torch.no_grad():  
                                ####### gold captions
                                tokenized = tokenizer(ques, captions, padding='longest', return_tensors="pt")
                                tokenized_inputs_gold = tokenized["input_ids"]
                                attn_mask_gold = tokenized["attention_mask"]
                            #     outputs_gold = model_downstream( tokenized_inputs_gold, attn_mask_gold)
                            #     # breakpoint()
                            #     f1_score_gold, answers_gold = compute_score_batch(outputs_gold, label, tokenized_inputs_gold, answer, tokenizer, tokenized_inputs_gold)
                                
                            outputs_fine = model_downstream(inputs_fine, attn_mask_fine)
                            # outputs_base = model_downstream_base(inputs_fine, attn_mask_fine)
                            f1_score_tuned, answers_tuned = compute_score_batch(outputs_fine, labels_fine, inputs_fine, valid_ans, tokenizer, tokenized_inputs_gold)
                            # f1_score_base, answers_base = compute_score_batch_downstream(outputs_base, labels_fine, inputs_fine, answer, tokenizer, tokenized_inputs_gold)


                            sep_token = 2
                            sep_indices = (inputs_fine == sep_token).int().argmax(dim=1)
                            # breakpoint()
                            mask = torch.arange(inputs_fine.cpu().size(1)).expand(inputs_fine.cpu().size(0), -1) > sep_indices.cpu().unsqueeze(1)
                            # expanded_mask = mask.unsqueeze(-1).expand(-1, -1, outputs.size(2))
                            mask = mask.to("cuda")
                            # outputs = expanded_mask*outputs
                            outputs_fine = outputs_fine.permute(0,2,1) # Permute for Cross Entropy loss
                            loss = loss_fn(outputs_fine, labels_fine)
                            loss = loss * mask # Mask out question tokens for loss
                            loss = loss.mean()
                            # loss = outputs.loss
                            this_loss = loss.item()
                            running_loss += this_loss
                            #loss = loss_fn(outputs, labels.float())
                            loss.backward()
                            optimizer2.step()
                            optimizer2.zero_grad()

                            inputs_fine = []
                            labels_fine = []
                            img_ids_fine = []
                            captions_fine = []
                            attn_mask_fine = []
                            labels_gold = []
                            valid_ans = []
        
                        # Downstream performance with greedy decoded caption


                        # if it%100==0:
                        #     #  
                        #     # print(base_score_c)
                        #     print(recap_score_c)
                        #     # print("####################ques#######################")
                        #     ques = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), ques)}
                        #     print(ques)
                        #     # print("####################base#######################")
                        #     # print(generated_caption_dict_base)
                        #     print("####################GT#######################")
                        #     caption_dict = {key: [value] for key,value in zip(image_id[0], captions)}
                        #     print(caption_dict)
                        #     # print("####################finetuned#######################")
                        #     # caption_dict_fine = {key: [value] for key,value in zip(img_ids_fine, cap_fine)}
                        #     # print(caption_dict)
                        #     print("###################recap########################")
                        #     print(generated_caption_baseline_dict)
                        #     print("####################answer#####################")
                        #     answers = [list(set(ans)) for ans in answer]
                        #     ans = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), answers)}
                        #     print(ans)
                        #     print("####################answer gold#####################")
                        #     ans_gold = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), answers_gold)}
                        #     print(ans_gold)
                        #     print("####################answer_recap#####################")
                        #     # breakpoint()
                        #     answers_tuned_dict = {key: value for key, value in zip(img_ids_fine, answers_tuned)}
                        #     print(answers_tuned_dict)
                        #     print("####################answer_base#####################")
                        #     answers_base_dict = {key: value for key, value in zip(img_ids_fine, answers_base)}
                        #     print(answers_base_dict)

                        #     print("####################scores_finetuned#####################")
                        #     scores_fine = {key: value for key, value in zip(img_ids_fine, f1_score_tuned)}
                        #     print(scores_fine)
                        #     print("####################scores_base#####################")
                        #     scores_base = {key: value for key, value in zip(img_ids_fine, f1_score_base)}
                        #     print(scores_base)
                        #     print("####################scores GT#####################")
                        #     scores_gt = {key: value for key, value in zip(generated_caption_baseline_dict.keys(), f1_score_gold)}
                        #     print(scores_gt)
                        #     print("###################sampling########################")
                        #     print(generated_caption_random_dict_base)
                        #     try: 
                        #         torch.save(model_downstream.module.state_dict(), 'Downstream_vqa_BIO_fine_contrastive_new_QA_moto_further.pth')
                        #     except:
                        #          torch.save(model_downstream.module.state_dict(), 'Downstream_vqa_BIO_fine_contrastive_new_QA_moto_further' +str(it)+'.pth')

                            # print(loss_r)
                        
                            # s = evaluation(model_captioning, coco_cider_dataloader, ref_caps)
                        # if it%250==0:
                        #         torch.save(model_captioning.module.state_dict(), './recaptioning_expon_it_'+str(it)+'.pth')
                        # pbar.set_postfix(loss=this_loss)
                        pbar.update()
                        # seq_recap = [len(x.split(' ')) for x in generated_caption_baseline]
                        # seq_base = [len(x.split(' ')) for x in captions]

                        # avg_seq_recap = np.mean(seq_recap)
                        # avg_seq_base = np.mean(seq_base)
                        # net_avg_seq = avg_seq_recap - avg_seq_base
                        # logging_step += 1
                        # wandb.log({"recall": recall_recap.mean()})
                        # wandb.log({"n-gram penality": np.array(distinct).mean()})
                        # wandb.log({"f-score-base": f1_score_base.mean()})
                        # wandb.log({"f-score-tuned": f1_score_tuned.mean()})
                        # wandb.log({"f-score-gold": f1_score_gold.mean()})
                        # wandb.log({"cider_recap": recap_score_c})
                        # wandb.log({"cider_base": base_score_c})
                        # wandb.log({"reward": reward.mean()})
                        # wandb.log({"loss": this_loss})
                        # wandb.log({"average length": avg_seq_recap})
                        # wandb.log({"net average length": net_avg_seq})
                        # wandb.log({"epoch": i})
                        # wandb.log({"step": logging_step})
                        # wandb.log({"running loss": (running_loss1/ (it + 1))})




# labels = []
# for  it,(label, mask, captions) in  enumerate(things_dataloader_train):
#     labels.append(label)
# z = sum(labels)
# bot = len(stuff_objects_coco_train)

# freq = sum(z)
# bot2 = sum(freq)
# weights =1-freq/bot2
import warnings
# warnings.filterwarnings("ignore")

from huggingface_hub import login
if __name__ == "__main__":
    SEED = 101
    torch.manual_seed(SEED)
    # os.environ["NCCL_DEBUG"] = "INFO"

    print("Custom params being set")
    parser = argparse.ArgumentParser(description="Example script with a boolean flag.")

    # Hyperparams and other
    NOTES = None # "No baseline"    # Notes to pass for wandb experiment
    # EXP_NAME = " 10k contrastive sampling, best settings, finetune downstream every 250it on 30k samples(train)"    # Name of the experiment
    # EXP_NAME = "image similarity AS OBJECTIVE, using mean of 3 patches/query for sim, k=5, alpha=0.6, adding scaling facotr X15" 
    EXP_NAME = " pali monitor" 

    LR = 1e-7
    LR_DOWN = 1e-7
    WANDB_STATUS = "online"
    stemmer = PorterStemmer()
    batch_size = 8
    
    # Adding a boolean flag --flag
    parser.add_argument(
        '--jh',
        action='store_true',
        help='Passing flag will set it to True, else False'
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Passing flag will set it to True, else False'
    )

    parser.add_argument(
        '--no_baseline',
        action='store_true',
        help='Passing flag will set it to True, else False'
    )
    
    args = parser.parse_args()
    if args.jh:
        print("############### Using JH's WandB key ###############")
        WANDB_KEY = "9ceeb6265e89c436c4dc698ef0e9b334039e42ef"
        print(f"~~~~~~~~~~~~~ {EXP_NAME} ~~~~~~~~~~~~~")
    else:
        print("############### Using Abdo's WandB key ###############")
        WANDB_KEY = "ee6091224cb7bb0fda72ab4cd492e55463c4813b"
        print(f"~~~~~~~~~~~~~ {EXP_NAME} ~~~~~~~~~~~~~")

    if args.no_wandb:
        print("~~~~~~~~~~~~~ No WandB. Hope you are just debugging!!! ~~~~~~~~~~~~~")
        WANDB_STATUS = "disabled"


    # labels = []
    # for  it,(label, mask, captions) in  enumerate(things_dataloader_train):
    #     labels.append(label)
    # z = sum(labels)
    # bot = len(stuff_objects_coco_train)

    # freq = sum(z)
    # bot2 = sum(freq)
    # weights =1-freq/bot2


    # coco_train = Coco_Dataset()
    # coco_val = Coco_Dataset(split = "val")
    # stuff_objects_train = Coco_Dataset_things()
    # stuff_objects_coco_train = Coco_Dataset_things(downstream= False)#with image
    # stuff_objects_coco_val = Coco_Dataset_things(split="val", downstream= False)#with image
    # stuff_objects_val = Coco_Dataset_things(split = "val")

    vqa_train = Coco_Dataset_VQA()
    vqa_val = Coco_Dataset_VQA(split = "val")
    # # Create shuffled indices
    # indices = torch.randperm(len(stuff_objects_train)).tolist()

    # # Create subsets from the datasets
    # coco_train_shuffeled = Subset(coco_train, indices)
    # stuff_objects_train_shuffeled = Subset(stuff_objects_train, indices)

    model_blip = blip_recaption()
    model_down = bert_recaption()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_blip = model_blip.to(device)
    # model_blip = DDP(model_blip)

    model_down = model_down.to(device)
    # model_down = DDP(model_down)
    for name, param in model_down.named_parameters():
            param.requires_grad = True

    # for name, param in model_down.module.named_parameters():
    #     if "model" in name or "Bert" in name and "vqa" not in name and "pooler" not in name:
    #         param.requires_grad = False

    # for name, param in model_down.module.model.encoder.layer[-1].named_parameters():
    #     param.requires_grad = True
    # for name, param in model_down.module.model.encoder.layer[-2].named_parameters():
    #     param.requires_grad = True
    # for name, param in model_down.module.model.encoder.layer[-3].named_parameters():
    #     param.requires_grad = True
    # for name, param in model_down.module.vqa.named_parameters():
    #     param.requires_grad = True

    optimizer1 = torch.optim.AdamW(model_blip.parameters(), lr=LR)
    optimizer2 = torch.optim.Adam(model_down.parameters(), lr=LR_DOWN)
    # params = list(model_blip.parameters()) + list(model_down.parameters())
    optimizer_pretraining = torch.optim.Adam(model_down.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pretraining, 'min', patience=5, factor=0.5)
    # optimizer3 = torch.optim.Adam(params, lr=0.0001)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #coco_dataloader_train = DataLoader(coco_train, batch_size=4,shuffle=True, collate_fn = coco_train.collate_fn, num_workers=1,
    #                                    drop_last=True)
    #coco_dataloader_val = DataLoader(coco_val, batch_size=32,shuffle=False, collate_fn = coco_train.collate_fn, num_workers=1,
    #                                    drop_last=True)

    coco_train = Coco_Dataset(split = "train")
    coco_val = Coco_Dataset(split = "val")
    coco_dataloader_train = DataLoader(coco_train, batch_size=batch_size,shuffle=False, collate_fn = coco_train.collate_fn, num_workers=1,
                                    drop_last=True)
    coco_dataloader_val = DataLoader(coco_val, batch_size=batch_size,shuffle=False, collate_fn = coco_val.collate_fn, num_workers=1,
                                    drop_last=True)

    # things_dataloader_train = DataLoader(stuff_objects_train, batch_size=512,shuffle=True, num_workers=4,collate_fn = stuff_objects_train.collate_fn,
    #                                     drop_last=True) #no image
    # things_coco_dataloader_train = DataLoader(stuff_objects_coco_train, batch_size=12,shuffle=True, num_workers=4,collate_fn = stuff_objects_coco_train.collate_fn,
    #                                     drop_last=True) #with image
    # things_coco_dataloader_val = DataLoader(stuff_objects_coco_val, batch_size=24,shuffle=False, num_workers=4,collate_fn = stuff_objects_coco_val.collate_fn,
    #                                     drop_last=True)

    vqa_dataloader_train = DataLoader(vqa_train, batch_size=32,shuffle=True,collate_fn = vqa_train.collate_fn,
                                        drop_last=True) #with image
    vqa_dataloader_val = DataLoader(vqa_val, batch_size=32,shuffle=False, collate_fn = vqa_val.collate_fn,
                                        drop_last=True)


    # vqa_train_rec = FineCapEval(split="train", downstream = False)
    # vqa_val_rec = FineCapEval(split = "val", downstream= False)
    vqa_train_rec = Coco_Dataset_VQA(split="train", downstream = False)
    vqa_val_rec = Coco_Dataset_VQA(split = "val", downstream= False)
    vqa_train_dataloader_rec = DataLoader(vqa_train_rec, batch_size=batch_size, shuffle=True, num_workers=1,collate_fn = vqa_train_rec.collate_fn,
                                        drop_last=False) #with image

    vqa_val_dataloader_rec = DataLoader(vqa_val_rec, batch_size=32,shuffle=False, num_workers=1,collate_fn = vqa_val_rec.collate_fn,
                                        drop_last=False)
    
    # model, optimizer, dataloader_train = accelerator.prepare(
    #      model, optimizer, dataloader_train
    #  )
    import warnings
    # warnings.filterwarnings("ignore")
    # fastmodel = fasttext.load_model('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/cc.en.300.bin')
    # fasttext.util.reduce_model(fastmodel, 50)
    # model_captioning = blip_recaption()
    # model_qa = bert_recaption()
    # model_downstream_state_dict = torch.load('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/Downstream_vqa_BIO_new.pth')
    # model_qa.load_state_dict(model_downstream_state_dict)
    # model_captioning_state_dict = torch.load('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/recaptioning_contrastive_1e-7,it21.pth', map_location='cpu')
    # model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
    # model_captioning = model_captioning.to('cuda:0')
    # model_qa = model_qa.to('cuda:0')
    # processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")



    # recap_validation(model_captioning, processor, model_qa, vqa_val_dataloader_rec)
    recaptioning(vqa_train_dataloader_rec, vqa_val_dataloader_rec, coco_dataloader_val, ref_caps, model_blip, model_down, optimizer1, optimizer2, None)
    # finetune_down(vqa_train_dataloader_rec, model_captioning, model_qa, optimizer1, optimizer2, None)

# #evaluation(model_blip, coco_dataloader_val, ref_caps)
# #recaptioning(things_coco_dataloader_train,stuff_objects_coco_val,ref_caps, model_blip, model_down, optimizer1, optimizer2 )
    # pretraining_down(vqa_dataloader_train, vqa_dataloader_val, model_down, optimizer_pretraining, None, scheduler)







# from recaption_model import blip_recaption, bert_recaption
# from recap_dataset import Coco_Dataset, Coco_Dataset_things
# from torch.utils.data import DataLoader, Subset
# import torch
# from tqdm import tqdm
# from accelerate import Accelerator
# import numpy as np
# import json
# from collections import defaultdict
# from pycocoevalcap.cider.cider import Cider
# import itertools
# from transformers import AutoProcessor, Blip2ForConditionalGeneration
# import torch.distributed as dist
# import torch.multiprocessing as mp
# # accelerator = Accelerator()

 

# coco_train = Coco_Dataset()
# stuff_objects_train = Coco_Dataset_things()



# # # Create shuffled indices
# # indices = torch.randperm(len(stuff_objects_train)).tolist()

# # # Create subsets from the datasets
# # coco_train_shuffeled = Subset(coco_train, indices)
# # stuff_objects_train_shuffeled = Subset(stuff_objects_train, indices)
# coco_val = Coco_Dataset(split = "val")
# batch_size = 32
# model_blip = blip_recaption()
# model_down = bert_recaption()

# for name, param in model_down.named_parameters():
#         param.requires_grad = True

# for name, param in model_down.named_parameters():
#     if "model" in name or "Bert" in name :
#         param.requires_grad = False


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
#     os.environ['MASTER_PORT'] = '12355'      # choose an open port
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
# def cleanup():
#     dist.destroy_process_group()




# optimizer = torch.optim.AdamW(model_blip.parameters(), lr=5e-5)
# optimizer2 = torch.optim.Adam(model_down.parameters(), lr=0.01)
# device = device if torch.cuda.is_available() else "cpu"
# coco_dataloader_train = DataLoader(coco_train, batch_size=4,shuffle=True, collate_fn = coco_train.collate_fn, num_workers=1,
#                                     drop_last=True)
# coco_dataloader_val = DataLoader(coco_val, batch_size=4,shuffle=False, collate_fn = coco_train.collate_fn, num_workers=1,
#                                     drop_last=True)

# # things_dataloader_train = DataLoader(stuff_objects_train, batch_size=512,shuffle=True, num_workers=1,collate_fn = stuff_objects_train.collate_fn,
# #                                     drop_last=True)

# # model, optimizer, dataloader_train = accelerator.prepare(
# #      model, optimizer, dataloader_train
# #  )

# # 

# def train_captioning(model, dataloader_train, optimizer):
#     model = model.to(device)
#     for e in range (0, 100):
#         with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader_train)) as pbar:
#             for it, (images, captions, mask, prompts, ids) in enumerate(dataloader_train):

            
#                 images, captions, prompts = images.to(device), captions.to(device), prompts.to(device)

                
#                 outputs = model(images,captions,prompts, ids)
                            
            
#                 loss = outputs.loss
                
#                 print("Loss:", loss.item())

#                 loss.backward()
#                 # accelerator.backward(loss)

#                 optimizer.step()
#                 optimizer.zero_grad()


#                 this_loss = loss.item()
#                 running_loss += this_loss

#                 pbar.set_postfix(loss=running_loss / (it + 1))
#                 pbar.update()

#                 loss = running_loss / len(dataloader)


# def load_references_from_json(file_path):

#     with open(file_path, 'r') as f:
#         data = json.load(f)
    
#     captions_by_id = defaultdict(list)
#     for item in data["annotations"]:
#         captions_by_id[item['image_id']].append(item['caption'])
    
#     return captions_by_id

# def evaluate_cider(gen_captions, ref_captions):
#     scorer = Cider()
#     cider_score, _ = scorer.compute_score(ref_captions, gen_captions)
#     print(f"CIDEr Score: {cider_score}")
#     return cider_score



# ref_caps = load_references_from_json("./annotations/captions_val2017.json")


# def evaluation(model, dataloader_val, ref_caps):

#     model.eval()

#     gen_caps = {}
#     with tqdm( unit='it', total=len(dataloader_val)) as pbar:
#         for it, (images, captions, mask, prompts, ids) in enumerate(dataloader_val):
#             images, captions, prompts = images.to(device), captions.to(device), prompts.to(device)

#             with torch.no_grad():
#                 output = model.generate(images,captions,prompts, ids)
#             gen_caps = {**gen_caps, **output}
#             pbar.update()
#     score = evaluate_cider(gen_caps, ref_caps)

# def pretraining_down(rank, world_size, model, optimizer):
#     setup(rank, world_size)
#     stuff_objects_train = Coco_Dataset_things()
#     sampler = torch.utils.data.distributed.DistributedSampler(stuff_objects_train, num_replicas=world_size, rank=rank)
#     things_dataloader_train = DataLoader(stuff_objects_train, sampler=sampler, batch_size=512,shuffle=True, num_workers=1,collate_fn = stuff_objects_train.collate_fn,
#                                         drop_last=True)



#     loss_fn = torch.nn.BCEWithLogitsLoss()
#     running_loss = 0.
#     #model = model.to(device)
#     device = torch.device(f'cuda:{rank}')
#     model.to(device)
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

#     with tqdm( unit='it', total=len(things_dataloader_train)) as pbar:
#         for  it,(labels,captions) in  enumerate(things_dataloader_train):
            
#             labels, captions = labels.to(device), captions.to(device)
#             outputs = model(captions)
#             #breakpoint()
#             loss = loss_fn(outputs, labels.float())
#             this_loss = loss.item()
#             running_loss += this_loss
#             optimizer2.step()
#             optimizer2.zero_grad()
#             pbar.set_postfix(loss=running_loss / (it + 1))
#             pbar.update()

#             loss = running_loss / len(things_dataloader_train)
#     cleanup()
# #pretraining_down(stuff_objects_train, model_down, optimizer2)
# world_size = 2
# mp.spawn(pretraining_down, args=(world_size, model_down, optimizer2), nprocs=world_size, join=True)




    #     outputs_binary = (torch.sigmoid(outputs) > threshold).int()
        
    #     #correct_predictions_batch = torch.sum(torch.eq(labels, outputs_binary), dim=1)
    #     non_zero_mask = labels != 0
    #     y_true_non_zero = labels[non_zero_mask]
    #     y_pred_non_zero = outputs_binary[non_zero_mask]
    
    # # Compute element-wise equality between non-zero true labels and predictions
    #     correct_predictions_non_zero = torch.sum(torch.eq(y_true_non_zero, y_pred_non_zero)).item()
    
    #     total_predictions = y_true_non_zero.size(0)
        
    #     accuracy = correct_predictions_non_zero / total_predictions
    #     # accuracy = torch.mean(accuracy_per_instance)
    #     print("##########Accuracy########: "+ str(accuracy))

