from recaption_model_vqa import blip_recaption, bert_recaption
from recap_dataset import Coco_Dataset, Coco_Dataset_things, Coco_Dataset_attributes, Coco_Dataset_Ref, Coco_Dataset_VQA_Comparision, Coco_Dataset_dummy
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
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer, AutoModelForVisualQuestionAnswering
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
from recap_dataset import Coco_Dataset, Coco_Dataset_things, Coco_Dataset_attributes, Coco_Dataset_Ref, Coco_Dataset_VQA_Comparision, FineCapEval, Coco_Dataset_VQA
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vqa_val_rec = FineCapEval(split = "val", downstream= False)
# vqa_train_rec = FineCapEval(downstream = False)
vqa_train_rec = Coco_Dataset_VQA(downstream = False)
vqa_val_rec = Coco_Dataset_VQA(split = "val", downstream= False)

vqa_train_dataloader_rec = DataLoader(vqa_train_rec, batch_size=32, shuffle=False, num_workers=1,collate_fn = vqa_train_rec.collate_fn,
                                    drop_last=True) #with image

vqa_val_dataloader_rec = DataLoader(vqa_val_rec, batch_size=32,shuffle=False, num_workers=1,collate_fn = vqa_val_rec.collate_fn,
                                    drop_last=True)



def evaluation(model, dataloader_val, ref_caps):

    model.eval()
    model = DDP(model.module)
    model = model.to(device)
    gen_caps = {}
    with tqdm( unit='it', total=len(dataloader_val)) as pbar:
        for it, (images, captions, mask, prompts, ids, label) in enumerate(dataloader_val):
            images, captions, prompts = images.to(device), captions.to(device), prompts.to(device)

            with torch.no_grad():
                # output = model.module.generate(images,captions,prompts, ids)
                processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
                generated_ids = model.module.model.generate(input_ids=prompts, pixel_values=images, max_length=60)
                generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
                output = {key: [value] for key,value in zip(ids[0], generated_caption)}

            gen_caps = {**gen_caps, **output}
            pbar.update()
    score = evaluate_cider(gen_caps, ref_caps)
    return score

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
  return norm

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
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


ref_caps = load_references_from_json("./my_datasets/annotations/captions_val2017.json")
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

                    exact_scores.append(compute_exact(answers_t, answers_p))
                    f1_scores.append(compute_f1(answers_t, answers_p))
                    # f1_macro = f1_score(target.cpu(), prediction.cpu(), average='macro')
                    # f_scores.append(f1_macro)


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
    return exact_score, f1_score, running_loss
from transformers import AutoProcessor, Blip2ForConditionalGeneration, Blip2Config, Blip2Model, Blip2Processor
    
from torchmetrics.multimodal.clip_score import CLIPScore
from functools import partial
def compute_score_batch(outputs, label, tokenized_input, answers, tokenizer, tokenized_inputs_gold):
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
                        answers_tokens = gold_tokens[ind_target: offseted_index+1] #get answer
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
                    answers_pred.append(answers_p)       
                    answers_t = " ".join(answers_t) #concat all the answers of one sample for f-score, not sure if there is a better way but I guess it works fine
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


# model_base = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
model_base = blip_recaption()
model_captioning = blip_recaption()
model_base_dict = torch.load('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/recaptioning_img_sim19.pth', map_location='cpu')
# model_captioning.load_state_dict(model_base_dict, strict=True)

# model_base = model_base.to("cuda:0")

# model_captioning = DDP(model_captioning, device_ids=[0])
model_captioning = model_captioning.to("cuda:0")
# model_captioning_state_dict = torch.load('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/BIO/recaptioning_contrastive_20.pth', map_location='cpu')
# model_base.load_state_dict(model_captioning_state_dict, strict=True)
model_base = model_base.to("cuda:0")

# processor2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# processor2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# import numpy as np
# from huggingface_hub import hf_hub_download
# finetuned_weights_state_dict = torch.load(hf_hub_download(repo_id="sashakunitsyn/vlrm-blip2-opt-2.7b", filename="vlrm-blip2-opt-2.7b.pt"))
# model_captioning.load_state_dict(finetuned_weights_state_dict, strict=False)

# clip_score_fn = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14-336")
# scores_VLRM = []
# scores_ours = []
alpha = 0.9
beta = 0.5
gamma = 0.5
temerature = 1
top_k = 5
configs = {"temperature": temerature,
               "top_k": top_k,
               "alpha": alpha,
               "beta" : beta,
               "gamma": gamma,
               "legend": "alpha the balancing between prob and contrastive, beta image contribution, gamma context contribution"
               
               }
WANDB_KEY = "ee6091224cb7bb0fda72ab4cd492e55463c4813b"
WANDB_STATUS = "online" #disabled, online, offline
# WANDB_STATUS = "disabled" #disabled, online, offline
# WANDB_STATUS = "offline" #disabled, online, offline
EXP_NAME = "entropy, image similarity reward" 
os.environ["WANDB_API_KEY"] = WANDB_KEY

def calculate_entropy(probabilities):
    """
    Calculate entropy for a single probability distribution (one timestep).
    Arguments:
    probabilities -- Tensor of shape (vocabulary_size), softmax output for a single timestep.
    
    Returns:
    entropy -- Scalar, the entropy value for that timestep.
    """
    # Ensure no log(0) by adding a small epsilon
    epsilon = 1e-9
    entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon))
    return entropy

with wandb.init(project="recap", mode=WANDB_STATUS,  name=EXP_NAME, config= configs):
    for  it,(label, answer, captions, ques, images, prompts, image_id) in  enumerate(vqa_train_dataloader_rec):
                        
                    # with torch.autograd.set_detect_anomaly(True):
            label, images2, prompts2, images, prompts = label.to(device), images.to(device), prompts.to(device), images.to(device), prompts.to(device)
            with torch.no_grad():
                            ## Greedy decode captioning model
                    generated_tokens = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images,  max_length=60, output_logits= True, return_dict_in_generate=True )
                    generated_caption = processor.batch_decode(generated_tokens[0], skip_special_tokens=True)
                    ## Greedy decode base captioning model
                    # generated_tokens_greedy_base = model_base.model_blip.generate(input_ids=prompts2, pixel_values=images2, max_length=60, output_logits= True, return_dict_in_generate=True )
                    # generated_caption_base = processor2.batch_decode(generated_tokens_greedy_base[0], skip_special_tokens=True)
                    z = [len(x.split(' ')) for x in generated_caption]
                    generated_caption_recap_dict = {key: [value] for key,value in zip(image_id[0], generated_caption)}
                    # generated_caption_dict_base = {key: [value] for key,value in zip(image_id[0], generated_caption_base)}
                    logits_recap = generated_tokens[1]
                    logits_recap = torch.stack(logits_recap, dim=1)
                    # logits_base = generated_tokens_greedy_base[1]
                    # logits_base = torch.stack(logits_base, dim=1)
                    entropies = []
                    # for timestep_probs in logits_base:
                    #     entropy = calculate_entropy(F.softmax(timestep_probs, dim=-1))
                    #     entropies.append(entropy.item())

                    # # Average entropy across all timesteps
                    # average_entropy_base = sum(entropies) / len(entropies)
                    for timestep_probs in logits_recap:
                        entropy = calculate_entropy(F.softmax(timestep_probs, dim=-1))
                        entropies.append(entropy.item())
                    average_entropy_recap = sum(entropies) / len(entropies)
                    
                    # probs_recap = torch.nn.functional.softmax(logits_recap, dim=-1)
                    # probs_base = torch.nn.functional.softmax(logits_base, dim=-1)
                    # wandb.log({"base entropy": average_entropy_base})
                    wandb.log({"recap entropy": average_entropy_recap})

                    # generated_tokens_beam = model_captioning.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=60,num_beams= 5)
                    # generated_tokens_beam_base = model_base.generate(input_ids=prompts2, pixel_values=images2, max_length=60,num_beams= 5)
                    # generated_caption_beam = processor.batch_decode(generated_tokens_beam, skip_special_tokens=True)
                    # generated_caption_beam_base = processor2.batch_decode(generated_tokens_beam_base, skip_special_tokens=True)

                    # generated_caption_recap_dict_beam = {key: [value] for key,value in zip(image_id[0], generated_caption_beam)}
                    # generated_caption_dict_base_beam = {key: [value] for key,value in zip(image_id[0], generated_caption_beam_base)}

                    print("############################################################################")
                    # print("###################base########################")
                    # print(generated_caption_dict_base)
                    print("###################blip base########################")
                    print(generated_caption_recap_dict)
                    # print("###################beam base########################")
                    # print(generated_caption_dict_base_beam)
                    # print("###################beam coco########################")
                    # print(generated_caption_recap_dict_beam)
                    # print("############################################################################")



#                     clip_score = clip_score_fn(clip, generated_caption_beam_base).detach()
#                     scores_ours.append(clip_score)
#                     clip_score = clip_score_fn(clip, generated_caption_beam).detach()
#                     scores_VLRM.append(clip_score)
# print(np.mean(scores_ours))
# print(np.mean(scores_VLRM))

                    

                    
