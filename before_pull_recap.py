from recaption_model_vqa import blip_recaption, bert_recaption
from recap_dataset import Coco_Dataset, Coco_Dataset_things, Coco_Dataset_attributes, Coco_Dataset_Ref, Coco_Dataset_VQA, Coco_Dataset_dummy
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
import evaluate


import os
import wandb
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
from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer
from torch.nn import DataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
# accelerator = Accelerator()

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

batch_size = 32
model_blip = blip_recaption()
model_down = bert_recaption()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_blip = model_blip.to(device)
# model_blip = DDP(model_blip)

model_down = model_down.to(device)
model_down = DDP(model_down)
for name, param in model_down.module.named_parameters():
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

optimizer1 = torch.optim.AdamW(model_blip.parameters(), lr=5e-6)
optimizer2 = torch.optim.SGD(model_down.module.parameters(), lr=0.0001)
# params = list(model_blip.parameters()) + list(model_down.parameters())
optimizer_pretraining = torch.optim.Adam(model_down.module.parameters(), lr=5e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pretraining, 'min', patience = 5, factor = 0.5)
# optimizer3 = torch.optim.Adam(params, lr=0.0001)
device = "cuda" if torch.cuda.is_available() else "cpu"
#coco_dataloader_train = DataLoader(coco_train, batch_size=4,shuffle=True, collate_fn = coco_train.collate_fn, num_workers=1,
#                                    drop_last=True)
#coco_dataloader_val = DataLoader(coco_val, batch_size=32,shuffle=False, collate_fn = coco_train.collate_fn, num_workers=1,
#                                    drop_last=True)

coco_train = Coco_Dataset()
coco_val = Coco_Dataset(split = "val")
coco_dataloader_train = DataLoader(coco_train, batch_size=16,shuffle=True, collate_fn = coco_train.collate_fn, num_workers=1,
                                   drop_last=True)
coco_dataloader_val = DataLoader(coco_val, batch_size=16,shuffle=False, collate_fn = coco_val.collate_fn, num_workers=1,
                                   drop_last=True)

# things_dataloader_train = DataLoader(stuff_objects_train, batch_size=512,shuffle=True, num_workers=4,collate_fn = stuff_objects_train.collate_fn,
#                                     drop_last=True) #no image
# things_coco_dataloader_train = DataLoader(stuff_objects_coco_train, batch_size=12,shuffle=True, num_workers=4,collate_fn = stuff_objects_coco_train.collate_fn,
#                                     drop_last=True) #with image
# things_coco_dataloader_val = DataLoader(stuff_objects_coco_val, batch_size=24,shuffle=False, num_workers=4,collate_fn = stuff_objects_coco_val.collate_fn,
#                                     drop_last=True)

vqa_dataloader_train = DataLoader(vqa_train, batch_size=256,shuffle=True,collate_fn = vqa_train.collate_fn,
                                    drop_last=True) #with image
vqa_dataloader_val = DataLoader(vqa_val, batch_size=128,shuffle=False, collate_fn = vqa_val.collate_fn,
                                    drop_last=True)


vqa_train_rec = Coco_Dataset_VQA(downstream = False)
vqa_val_rec = Coco_Dataset_VQA(split = "val", downstream= False)
vqa_train_dataloader_rec = DataLoader(vqa_train_rec, batch_size=20,shuffle=True, num_workers=1,collate_fn = vqa_train_rec.collate_fn,
                                    drop_last=True) #with image

vqa_val_dataloader_rec = DataLoader(vqa_val_rec, batch_size=16,shuffle=False, num_workers=1,collate_fn = vqa_val_rec.collate_fn,
                                    drop_last=True)
# model, optimizer, dataloader_train = accelerator.prepare(
#      model, optimizer, dataloader_train
#  )

# 

def train_captioning(model, dataloader_train, optimizer):
    model = model.to(device)
    for e in range (0, 100):
        with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader_train)) as pbar:
            for it, (images, captions, mask, prompts, ids) in enumerate(dataloader_train):

            
                images, captions, prompts = images.to(device), captions.to(device), prompts.to(device)

                
                outputs = model(images,captions,prompts, ids)
                
                loss = outputs.loss
                
                print("Loss:", loss.item())

                loss.backward()
                # accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()


                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

                loss = running_loss / len(dataloader_train)

                # if val_loss < best_val:
                #  best_val = val_loss
                # checkpoint = {
                #     'epoch': epoch + 1,
                #     'model_state_dict': model.module.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': epoch_loss
                # }
                # torch.save(checkpoint, 'blip2.pth')



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
    print(f"CIDEr Score: {cider_score}")
    return cider_score
def evaluate_rouge(ref_captions, gen_captions):
    scorer = Rouge()
    rouge_score, _ = scorer.compute_score(gen_captions, ref_captions)
    print(f"rouge Score: {rouge_score}")
    return rouge_score
def evaluate_spice(gen_captions, ref_captions):

    os.environ['CORENLP'] = '/home/israfel.salazar/abdo/Recaption/stanford-corenlp-4.5.7'
    
    scorer = Spice()
    spice_score, _ = scorer.compute_score(ref_captions, gen_captions)
    print(f"SPICE Score: {spice_score}")
    return spice_score


ref_caps = load_references_from_json("./my_datasets/annotations/captions_val2017.json")
ref_caps_train = load_references_from_json("./my_datasets/annotations/captions_train2017.json")

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


import re
import string
import collections

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
  return normalize_answer(s).split()

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
    # breakpoint()
    return f1


def validation_down(things_dataloader_val, model):
    model.eval()
    with tqdm( unit='it', total=len(things_dataloader_val)) as pbar:
        threshold = 0.5
        exact_scores = []
        f1_scores = []
        running_loss = 0.0
        with torch.no_grad():
            for  it, (span_start, span_end, tokenized_input, attn_mask, answer, caption, ques) in  enumerate(things_dataloader_val):
                span_start, span_end, tokenized_input, attn_mask = span_start.to(device), span_end.to(device), tokenized_input.to(device), attn_mask.to(device)
                tokenizer = things_dataloader_val.dataset.tokenizer
                outputs = model(span_start, span_end, tokenized_input, attn_mask)
                start_logits = outputs.start_logits.argmax(dim=1)
                end_logits = outputs.end_logits.argmax(dim=1)
                running_loss += outputs.loss.mean().item()

                for i, (st_idx, end_idx) in enumerate(zip(start_logits, end_logits)):
                    pred_tokens = tokenized_input[i][st_idx:end_idx+1]
                    pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    ans_toks = tokenized_input[i][span_start[i]:span_end[i]+1]
                    ans_text = tokenizer.decode(ans_toks, skip_special_tokens=True)
                
                    exact_scores.append(compute_exact(ans_text, pred_text))
                    f1_scores.append(compute_f1(ans_text, pred_text))
        exact_scores = torch.tensor(exact_scores).type(torch.float)
        f1_scores = torch.tensor(f1_scores).type(torch.float)
        exact_score = 100. * exact_scores.mean().item()
        f1_score = 100. * f1_scores.mean().item()
        running_loss = running_loss / len(things_dataloader_val)
    model.train()
    return exact_score, f1_score, running_loss

def compute_score_batch(outputs, span_start, span_end, tokenized_input, tokenizer):

    start_logits = outputs.start_logits.argmax(dim=1)
    end_logits = outputs.end_logits.argmax(dim=1)
    running_loss = outputs.loss.mean().item()
    exact_scores = []
    f1_scores = []

    for i, (st_idx, end_idx) in enumerate(zip(start_logits, end_logits)):
        pred_tokens = tokenized_input[i][st_idx:end_idx+1]
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        ans_toks = tokenized_input[i][span_start[i]:span_end[i]+1]
        ans_text = tokenizer.decode(ans_toks, skip_special_tokens=True)
    
        exact_scores.append(compute_exact(ans_text, pred_text))
        f1_scores.append(compute_f1(ans_text, pred_text))
    exact_scores = torch.tensor(exact_scores).type(torch.float)
    f1_scores = torch.tensor(f1_scores).type(torch.float)
    exact_score =  exact_scores.mean().item()

    f1_score = f1_scores.mean().item()
    return exact_score, f1_scores

def recaptioning(things_coco_dataloader_train, coco_dataloader_val, coco_cider_dataloader, ref_caps, model_captioning, model_downstream, optimizer1, optimizer2, weights):

    loss_fn = torch.nn.BCEWithLogitsLoss()
    model_downstream_state_dict = torch.load('Downstream_vqa_span.pth')
    model_downstream.module.load_state_dict(model_downstream_state_dict)
    os.environ["WANDB_API_KEY"] = "ee6091224cb7bb0fda72ab4cd492e55463c4813b"

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    #model_captioning_state_dict = torch.load('../recaptioning_it_750.pth', map_location='cpu')
    #model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
    model_downstream.module.load_state_dict(model_downstream_state_dict, strict=True)
    for name, param in model_captioning.model_blip.named_parameters():
        param.requires_grad = False
    for name, param in model_captioning.model_blip.qformer.named_parameters():
        param.requires_grad = True
    for name, param in model_captioning.model_blip.language_projection.named_parameters():
        param.requires_grad = True


    model_downstream = DDP(model_downstream.module)
    model_downstream = model_downstream.to("cuda")
    for name, param in model_downstream.module.named_parameters():
          param.requires_grad = False
    model_captioning = DDP(model_captioning, device_ids=[1])
    model_captioning = model_captioning.to("cuda:1")
    device = "cuda:1"
    model_base = blip_recaption()
    model_base = DDP(model_base)
    model_base = model_base.to("cuda:0")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    best_val = 10000
    config = {"name": "0.25 baseline, only reward"}
    with wandb.init(project="recap", config=config):
        # wandb.watch(model_captioning,log="all", log_freq=1)
        for i in range(0,1):
            running_loss1 = 0.
            running_loss2 = 0.
            # T = validation_down(coco_dataloader_val, model_downstream)

            # s,s2 = evaluation(model_captioning, coco_cider_dataloader, ref_caps)
            # breakpoint()

            # print("##########CIDEr########: "+ str(s))
            # print("##########F1-Score########: "+ str(f_score))
            with tqdm( unit='it', total=len(things_coco_dataloader_train)) as pbar:

                for  it,(span_start, span_end, answer, captions, ques, images, prompts, image_id) in  enumerate(things_coco_dataloader_train):
                    # with torch.autograd.set_detect_anomaly(True):
                        span_start, span_end, images, prompts = span_start.to(device), span_end.to(device), images.to(device), prompts.to(device)
                        images_base, prompts_base = images.to("cuda:0"),prompts.to("cuda:0")
                        
                        with torch.no_grad():
                        
                            generated_tokens_greedy = model_captioning.module.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=50)
                            generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True)
                            generated_tokens_greedy_base = model_base.module.model_blip.generate(input_ids=prompts_base, pixel_values=images_base, max_length=50)
                            generated_caption_baseline_base = processor.batch_decode(generated_tokens_greedy_base, skip_special_tokens=True)
                            z = [len(x.split(' ')) for x in generated_caption_baseline]
                            generated_caption_baseline_dict = {key: [value] for key,value in zip(image_id[0], generated_caption_baseline)}
                            generated_caption_baseline_dict_base = {key: [value] for key,value in zip(image_id[0], generated_caption_baseline_base)}

                        # caption_dict = {key: [value] for key,value in zip(image_id[0], gt_caption)}
                            caption_dict = {key: ref_caps_train[key] for key in image_id[0] }
                            # breakpoint()
                            base_score_r = evaluate_rouge(generated_caption_baseline_dict_base, caption_dict)
                            recap_score_r = evaluate_rouge(generated_caption_baseline_dict, caption_dict) #use ref caps since we are using dict
                            base_score_c = evaluate_cider(generated_caption_baseline_dict_base, caption_dict)
                            recap_score_c = evaluate_cider(generated_caption_baseline_dict, caption_dict) #use ref caps since we are using dict
                            

                        try:
                            outs  = model_captioning.module.model_blip.generate(input_ids=prompts, pixel_values=images, do_sample=True, max_length=50,output_logits= True, return_dict_in_generate=True  )
                            flag = 0
                        except:
                            print("cuda out of memory")
                            flag = 1
                            continue
                        generated_tokens_sampling = outs[0]
                        logits = outs[1]
                        logits = torch.stack(logits, dim=1)
                        
                        indices = generated_tokens_sampling.unsqueeze(-1)
                        words_prob = torch.gather(logits, dim=2, index=indices).squeeze(-1)

                        # # print(generated_tokens_sampling)
                        generated_caption = processor.batch_decode(generated_tokens_sampling, skip_special_tokens=True) 
                        
                        generated_caption_random_dict = {key: [value] for key,value in zip(image_id[0], generated_caption)}
                        generated_caption_random_dict_base = {key: [value] for key,value in zip(image_id[0], generated_caption)}

                            # #breakpoint()
                            


                        tokenized = tokenizer(ques, generated_caption, padding='longest', return_tensors="pt")
                        tokenized_inputs = tokenized["input_ids"]
                        attn_mask = tokenized["attention_mask"]

                        outputs = model_downstream(span_start, span_end, tokenized_inputs, attn_mask)
                        # breakpoint()
                        # loss_r = loss_fn(outputs, labels) #need accuracy
                        loss_r = outputs.loss
                        exact_score_r, f1_score_r = compute_score_batch(outputs, span_start, span_end, tokenized_inputs, tokenizer)
                        with torch.no_grad():  
                            tokenized = tokenizer(ques, generated_caption_baseline, padding='longest', return_tensors="pt")
                            tokenized_inputs = tokenized["input_ids"]
                            attn_mask = tokenized["attention_mask"]
                            outputs_baseline = model_downstream(span_start, span_end, tokenized_inputs, attn_mask)
                            exact_score_baseline, f1_score_baseline = compute_score_batch(outputs_baseline, span_start, span_end, tokenized_inputs, tokenizer)
                            loss_baseline = outputs_baseline.loss


#################################################################################
                        perplexity = evaluate.load("perplexity", module_type="metric")
                        # breakpoint()
                        prep_recaption = perplexity.compute(model_id='gpt2',add_start_token=False,predictions=generated_caption_baseline, device="cuda")
                        prep_base = perplexity.compute(model_id='gpt2',add_start_token=False,predictions=generated_caption_baseline_base, device="cuda")
################################################################################
                        prep = torch.tensor(prep_recaption["perplexities"]) - torch.tensor(prep_base["perplexities"])
                        reward = f1_score_r - 0.25

                        reward_anchor_r = recap_score_r - base_score_r
                        reward_anchor_c = recap_score_c - base_score_c
                        if recap_score_r <0.1:
                            break
                        # breakpoint() #check here
                        # breakpoint()
                        total_reward = reward + prep
                        print("downstream reward: ",reward)
                        # print("preplexity: ", prep.mean())
                        # print("CIDEr: ",reward_anchor_c)
                        loss = -torch.mean(words_prob, -1) * total_reward.to(device)
                        # breakpoint()
                        loss = loss.mean()
                        loss.backward()
                        # optimizer1.step()
                        # optimizer1.zero_grad()
                        

                        if it % 3 == 0 or flag==1:
                            optimizer1.step()
                            optimizer1.zero_grad()
                            torch.cuda.empty_cache()
                        # optimizer1.step()
                        this_loss = loss.item()
                        running_loss1 += this_loss

                        if it%25==0:
                            # breakpoint()
                            print(ques)
                            print(generated_caption_baseline_dict_base)
                            print(generated_caption_baseline_dict)
                            print(generated_caption_random_dict_base)

                            # print(loss_r)
                        
                            # s = evaluation(model_captioning, coco_cider_dataloader, ref_caps)
                        # if it%750==0:generated_caption_random_dict_base
                        #         torch.save(model_captioning.module.state_dict(), '/l/users/israfel.salazar/abdo/recaptioning_it_'+str(it)+'.pth')
                        pbar.set_postfix(loss=this_loss)
                        pbar.update()
                        seq = [len(x.split(' ')) for x in generated_caption_baseline]
                        avg_seq = np.mean(seq)
                        wandb.log({"Rouge reward ": reward_anchor_r/2})
                        wandb.log({"reward  ": total_reward.mean()})
                        wandb.log({"loss  ": this_loss})
                        wandb.log({"prep_recaption  ": prep_recaption})
                        wandb.log({"prep_base  ": prep_base})
                        wandb.log({"average length  ": avg_seq})
                        wandb.log({"f1_score  ": f1_score_r.mean()})



def pretraining_down(things_dataloader_train, things_dataloader_val, model, optimizer, weights, scheduler):
    # weights = weights.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    
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
            for  it,(span_start, span_end, tokenized_input, attn_mask, answer, caption, ques) in  enumerate(things_dataloader_train):
                span_start, span_end, tokenized_input, attn_mask = span_start.to(device), span_end.to(device), tokenized_input.to(device), attn_mask.to(device)
               
                outputs = model(span_start, span_end, tokenized_input, attn_mask)
                # breakpoint()
                # loss = loss_fn(outputs, labels)
                loss = outputs.loss
                this_loss = loss.item()
                running_loss += this_loss
                #loss = loss_fn(outputs, labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

                loss = running_loss / len(things_dataloader_train)
        exact_score, f1_score, val_loss = validation_down(things_dataloader_val, model)
        scheduler.step(val_loss)

        # if val_loss < best_val:
        #  patience = 0
        #  best_val = val_loss
        #  torch.save(model.module.state_dict(), 'Downstream_vqa_span.pth')
        # else:
        #     patience+=1
        # if patience ==50:
        #     break
        print("##########VAL LOSS########: "+ str(val_loss))
        print("##########Exact Score########: "+ str(exact_score))
        print("##########F1 Score########: "+ str(f1_score))
        # print("##########accuracy########: "+ str(acc))
        # print("##########accuracy2########: "+ str(acc2))

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


#pretraining_down(vqa_dataloader_train, vqa_dataloader_val, model_down, optimizer_pretraining, None, scheduler)
#evaluation(model_blip, coco_dataloader_val, ref_caps)
recaptioning(vqa_train_dataloader_rec, vqa_val_dataloader_rec, coco_dataloader_val, ref_caps, model_blip, model_down, optimizer1, optimizer2, None )







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