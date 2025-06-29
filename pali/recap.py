from recaption_model import blip_recaption, bert_recaption
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
        self.nll_loss = nn.BCEWithLogitsLoss(reduction='none')
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
                                                          weight=weights, reduction='none')            
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

coco_train = Coco_Dataset()
coco_val = Coco_Dataset(split = "val")
stuff_objects_train = Coco_Dataset_things()
stuff_objects_coco_train = Coco_Dataset_things(downstream= False)#with image
stuff_objects_coco_val = Coco_Dataset_things(split="val", downstream= False)#with image
stuff_objects_val = Coco_Dataset_things(split = "val")


# # Create shuffled indices
# indices = torch.randperm(len(stuff_objects_train)).tolist()

# # Create subsets from the datasets
# coco_train_shuffeled = Subset(coco_train, indices)
# stuff_objects_train_shuffeled = Subset(stuff_objects_train, indices)

batch_size = 32
model_blip = blip_recaption()
model_down = bert_recaption()
# model_blip = DDP(model_blip)
# model_blip = model_blip.to("cuda")


# model_down = model_down.to("cuda")
model_down = DDP(model_down, device_ids=[0])
for name, param in model_down.module.named_parameters():
        param.requires_grad = True

for name, param in model_down.module.named_parameters():
    if "model" in name or "Bert" in name :
        param.requires_grad = False
for name, param in model_blip.model_blip.named_parameters():
        param.requires_grad = False
for name, param in model_blip.model_blip.qformer.named_parameters():
        param.requires_grad = True

optimizer1 = torch.optim.AdamW(model_blip.model_blip.parameters(), lr=1e-5)
optimizer2 = torch.optim.SGD(model_down.module.parameters(), lr=0.0001)
params = list(model_blip.parameters()) + list(model_down.parameters())
optimizer_pretraining = torch.optim.Adam(model_down.module.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pretraining, 'min', patience = 5, factor = 0.5)
# optimizer3 = torch.optim.Adam(params, lr=0.0001)
device = "cuda" if torch.cuda.is_available() else "cpu"
coco_dataloader_train = DataLoader(coco_train, batch_size=24,shuffle=True, collate_fn = coco_train.collate_fn, num_workers=1,
                                   drop_last=True)
coco_dataloader_val = DataLoader(coco_val, batch_size=32,shuffle=False, collate_fn = coco_val.collate_fn, num_workers=1,
                                   drop_last=True)

things_dataloader_train = DataLoader(stuff_objects_train, batch_size=512,shuffle=True, num_workers=4,collate_fn = stuff_objects_train.collate_fn,
                                    drop_last=True) #no image

things_dataloader_val = DataLoader(stuff_objects_val, batch_size=256,shuffle=True, num_workers=4,collate_fn = stuff_objects_train.collate_fn,
                                    drop_last=True) #no image

things_coco_dataloader_train = DataLoader(stuff_objects_coco_train, batch_size=32,shuffle=True, num_workers=1,collate_fn = stuff_objects_coco_train.collate_fn,
                                    drop_last=True) #with image
things_coco_dataloader_val = DataLoader(stuff_objects_coco_val, batch_size=6,shuffle=False, num_workers=1,collate_fn = stuff_objects_coco_val.collate_fn,
                                    drop_last=True)
things_dataloader_train = DataLoader(stuff_objects_train, batch_size=512,shuffle=True, num_workers=4,collate_fn = stuff_objects_train.collate_fn,
                                    drop_last=True) 
vqa_train = Coco_Dataset_VQA(downstream = False)
vqa_val = Coco_Dataset_VQA(split = "val", downstream= False)
vqa_train_dataloader = DataLoader(vqa_train, batch_size=6,shuffle=True, num_workers=1,collate_fn = vqa_train.collate_fn,
                                    drop_last=True) #with image
vqa_val_dataloader = DataLoader(vqa_val, batch_size=6,shuffle=False, num_workers=1,collate_fn = vqa_val.collate_fn,
                                    drop_last=True)

def train_captioning(model, dataloader_train, optimizer):
    model = DDP(model.module)
    model = model.to("cuda")
    model.train()
    running_loss = 0.0
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
    for e in range (0, 100):
        with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader_train)) as pbar:
            for it, (images, captions, mask, prompts, ids) in enumerate(dataloader_train):

            
                images, captions, prompts = images.to(device), captions.to(device), prompts.to(device)
                # if it%100==0 :
                #     with torch.no_grad():             
                #         generated_tokens_greedy = model.module.model.generate(input_ids=prompts, pixel_values=images, max_length=30)
                #         generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True) 

                # try:
                outputs = model(pixel_values = images, labels = captions, input_ids = prompts)
                breakpoint()
                # except:
                #     continue
                
                loss = outputs.loss
                

                
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
    # model = DDP(model.module)
    # model = model.to("cuda")
    gen_caps = {}
    with tqdm( unit='it', total=len(dataloader_val)) as pbar:
        for it, (images, captions, mask, prompts, ids) in enumerate(dataloader_val):
            images, captions, prompts = images.to("cuda"), captions.to("cuda"), prompts.to("cuda")

            with torch.no_grad():
                # output = model.module.generate(images,captions,prompts, ids)
                processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
                with torch.no_grad():
                    generated_tokens = model.module.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=60)
                generated_caption = processor.batch_decode(generated_tokens, skip_special_tokens=True)
                output = {key: [value] for key,value in zip(ids[0], generated_caption)}

            gen_caps = {**gen_caps, **output}
            pbar.update()
    # breakpoint()
    ref_caps = dict(list(ref_caps.items())[0:len(gen_caps)])
    score2 = evaluate_rouge(gen_caps, ref_caps)
    score = evaluate_cider(gen_caps, ref_caps)
    
    return score, score2


def validation_down(things_dataloader_val, model):
        loss_fn = torch.nn.CrossEntropyLoss()
        running_loss = 0.
        y_pred = []
        y_true = []
        y_pred_args = []
        y_true_args = []
        with tqdm( unit='it', total=len(things_dataloader_val)) as pbar:
            threshold = 0.5
            with torch.no_grad():
                for  it,(labels, captions, ques) in  enumerate(things_dataloader_val):
                #for  it,(obj, subj, caption, mask, subj_c, obj_c) in  enumerate(things_dataloader_train):                
                    #labels, captions, mask = labels.to(device), captions.to(device), mask.to(device)
                    labels = labels.to(device)

                    outputs = model(captions, ques)
                    #outputs = model(captions, mask)
                    #outputs = model(obj, subj, caption, mask, subj_c, obj_c)
                    # breakpoint()
                    loss = loss_fn(outputs, labels)
                    this_loss = loss.item()
                    running_loss += this_loss
                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()
                    loss = running_loss / len(things_dataloader_val)
                   # outputs_binary = (torch.sigmoid(outputs) > threshold).int()
                    # breakpoint()
                    # preds = outputs.argmax(1)
                    # preds = F.one_hot(out_max, num_classes=200)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(outputs.cpu().tolist())
                    y_true_args.extend(torch.argmax(labels,1).cpu().tolist())
                    y_pred_args.extend(torch.argmax(outputs,1).cpu().tolist())
                # report = classification_report(y_true, y_pred, output_dict=True)
                # fscore = report['samples avg']['f1-score']
                acc2 = accuracy_score(y_true_args, y_pred_args)
               
                acc = (torch.argmax(torch.FloatTensor(y_pred), 1) == torch.argmax(torch.FloatTensor(y_true), 1)).float().mean()
                
                return loss, acc, acc2

from undecorated import undecorated
from types import MethodType
import os
import wandb
def recaptioning(things_coco_dataloader_train,coco_dataloader_val, coco_cider_dataloader, things_dataloader_val, ref_caps, model_captioning, model_downstream, optimizer1, optimizer2, weights):
    # loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    os.environ["WANDB_API_KEY"] = "ee6091224cb7bb0fda72ab4cd492e55463c4813b"

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    model_downstream_state_dict = torch.load('/home/israfel.salazar/abdo/Recaption/Downstream_vqa_really_what_only_augmented.pth')
    #model_captioning_state_dict = torch.load('../recaptioning_it_750.pth', map_location='cpu')
    #model_captioning.load_state_dict(model_captioning_state_dict, strict=True)
    model_downstream.module.load_state_dict(model_downstream_state_dict, strict=True)
    for name, param in model_captioning.model_blip.named_parameters():
        param.requires_grad = False
    for name, param in model_captioning.model_blip.qformer.named_parameters():
        param.requires_grad = True
    for name, param in model_captioning.model_blip.language_projection.named_parameters():
        param.requires_grad = True

    for name, param in model_downstream.module.named_parameters():
        if "model" in name or "Bert" in name :
          param.requires_grad = False
    # model_downstream = DDP(model_downstream.module)
    # model_downstream = model_downstream.to("cuda")
    model_captioning = DDP(model_captioning, device_ids=[0])
    # model_captioning = model_captioning.to("cuda:0")
    #model_base = blip_recaption()
    #model_base = DDP(model_base)
    #model_base = model_base.to("cuda:1")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
   # tokenizer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # tokenizer = AutoTokenizer.from_pretrained("efederici/sentence-bert-base")
    weights = weights.to("cuda")
    #loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = weights**3)
    # breakpoint()
    # generate_with_grad = undecorated(model_captioning.module.model.generate)
    # model_captioning.module.generate_with_grad = MethodType(generate_with_grad, model_captioning.module.model)
    loss_down = MultiClassBCELoss()
    best_val = 10000
    # cider_dict ={}
    # scorer = Cider(df="/home/israfel.salazar/abdo/Recaption/coco_caption/data/coco-val-df")
    # import time
    # for  it,(images, captions, mask, prompt_list, ids) in  enumerate(coco_dataloader_train):
    #         images, prompt_list = images.to("cuda"),prompt_list.to("cuda")
    #         with torch.no_grad():
    #                 generated_tokens_greedy = model_captioning.module.model_blip.generate(input_ids=prompt_list, pixel_values=images, max_length=60)
    #                 generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True)

    #                 generated_caption_baseline_dict = {key: [value] for key,value in zip(ids[0], generated_caption_baseline)}
    #                # caption_dict = {key: [value] for key,value in zip(image_id[0], gt_caption)}
    #                 caption_dict = {key: ref_caps_train[key] for key in ids[0] }
    #                 for gen, cap, key in zip(generated_caption_baseline_dict.items(), caption_dict.items(), generated_caption_baseline_dict.keys()):
    #                     s = time.time()
    #                     cider_score, _  = scorer.compute_score({cap[0]:cap[1]}, {gen[0]:gen[1]})
    #                     print(f"CIDEr Score: {cider_score}")
    #                     cider_dict[key]=cider_score
    #                     print(time.time()-s)
    #                     breakpoint()
    # breakpoint()
    with wandb.init(project="recap"):
        # wandb.watch(model_captioning,log="all", log_freq=1)
        for i in range(0,1):
            running_loss1 = 0.
            running_loss2 = 0.
            #_, f_score = validation_down(things_dataloader_val, model_downstream)

            s,s2 = evaluation(model_captioning, coco_cider_dataloader, ref_caps)
            breakpoint()

            # print("##########CIDEr########: "+ str(s))
            # print("##########F1-Score########: "+ str(f_score))
            with tqdm( unit='it', total=len(things_coco_dataloader_train)) as pbar:


                for  it,(images, ques, mask, prompts, image_id, labels, gt_caption) in  enumerate(things_coco_dataloader_train):
                
                    images_base, prompts_base, labels_base = images.to("cuda:1"),prompts.to("cuda:1"),labels.to("cuda:1")
                    images, prompts, labels = images.to("cuda:0"),prompts.to("cuda:0"),labels.to("cuda:0")
                    #z = model_captioning.module.Hope(pixel_values = images,input_ids =prompts)
                    
                    with torch.no_grad():
                    
                        generated_tokens_greedy = model_captioning.module.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=50)
                        generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True)
                        generated_tokens_greedy_base = model_base.module.model_blip.generate(input_ids=prompts_base, pixel_values=images_base, max_length=50)
                        generated_caption_baseline_base = processor.batch_decode(generated_tokens_greedy_base, skip_special_tokens=True)
                        # print(image_id)
                        # print(ques) 
                        # print("#############################################################")
                        # print(gt_caption)
                        # print("#############################################################")
                        # print(generated_caption_baseline)
                        generated_caption_baseline_dict = {key: [value] for key,value in zip(image_id[0], generated_caption_baseline)}
                        generated_caption_baseline_dict_base = {key: [value] for key,value in zip(image_id[0], generated_caption_baseline_base)}

                    # caption_dict = {key: [value] for key,value in zip(image_id[0], gt_caption)}
                        caption_dict = {key: ref_caps_train[key] for key in image_id[0] }
                        # breakpoint()
                        base_score_r = evaluate_rouge(generated_caption_baseline_dict_base, caption_dict)
                        recap_score_r = evaluate_rouge(generated_caption_baseline_dict, caption_dict) #use ref caps since we are using dict
                        base_score_c = evaluate_cider(generated_caption_baseline_dict_base, caption_dict)
                        recap_score_c = evaluate_cider(generated_caption_baseline_dict, caption_dict) #use ref caps since we are using dict
                        
                    #  score_before_s = evaluate_spice(generated_caption_baseline_dict, caption_dict)
                        
                    
                    
                    # if it%2 == 0:
                    #     continue
                    #     with torch.no_grad():
                    #         #breakpoint()
                            
                    #         outs = model_captioning.module.model.generate(input_ids=prompts, pixel_values=images, max_length=60,num_beams= 3, output_logits= True, return_dict_in_generate=True )
                            
                    #         # outs[0] generated tokens, outs[1] logits, outs[2] hidden state
                    #         generated_tokens_beam = outs[0]
                    #         logits = outs[1]
                    #         # logits = torch.stack(logits, dim=1)
                    #         # indices = generated_tokens_beam.unsqueeze(-1)
                    #         # words_prop = torch.gather(logits, dim=2, index=indices).squeeze(-1)
                    #         generated_caption = processor.batch_decode(generated_tokens_beam, skip_special_tokens=True) 
                    # else:

                            # z = model_captioning.module.Hope(pixel_values = images,input_ids =prompts)
                            # generated_tokens_sampling = z[1]
                            # words_prob = z[0]
                            # breakpoint()
                            # breakpoint()
                    try:
                        outs  = model_captioning.module.model_blip.generate(input_ids=prompts, pixel_values=images, do_sample=True, max_length=50,output_logits= True, return_dict_in_generate=True  )
                        flag = 0
                    except:
                        print("cuda thingy")
                        flag = 1
                        continue
                    generated_tokens_sampling = outs[0]
                    logits = outs[1]
                    logits = torch.stack(logits, dim=1)
                
                    indices = generated_tokens_sampling.unsqueeze(-1)
                    words_prob = torch.gather(logits, dim=2, index=indices).squeeze(-1)
                    

                    # # print(generated_tokens_sampling)
                    generated_caption = processor.batch_decode(generated_tokens_sampling, skip_special_tokens=True) 
                        
                        # #breakpoint()
                        


                    # if it%2==0:
                    #     continue
                    #     tokenized_caption_ = tokenizer(generated_caption,padding='longest', return_tensors='pt')
                    #     caption = tokenized_caption_["input_ids"]
                    #     mask = tokenized_caption_['attention_mask']
                    #     outputs = model_downstream(caption, mask)
                    #     #outputs = model(obj, subj, caption, mask, subj_c, obj_c)
                    #     # breakpoint()
                    #     weights = weights**3
                    #     loss = loss_down(outputs, labels.float(), weights.repeat(labels.size(0),1))
                    #     #loss = loss_fn(outputs, labels.float())
                    #     loss.backward()
                    #     this_loss = loss.item()
                    #     running_loss2 += this_loss        
                    #     optimizer2.step()
                    #     optimizer2.zero_grad()
                    # else:

                    #caption_baseline = [c + ' ' + q for c, q in zip(generated_caption_baseline, ques)]
                # tokenized_caption_baseline = tokenizer(generated_caption_baseline,padding=True, truncation=True, return_tensors='pt')
                    # caption_baseline = tokenized_caption_baseline["input_ids"]
                    # mask_baseline = tokenized_caption_baseline['attention_mask']
                    
                # caption_ = [c + ' ' + q for c, q in zip(generated_caption, ques)]
                # tokenized_caption_ = tokenizer(generated_caption,padding=True, truncation=True, return_tensors='pt')
                    # caption = tokenized_caption_["input_ids"]
                    # mask = tokenized_caption_['attention_mask']
            
                

                
                    outputs = model_downstream(generated_caption, ques)
                    # breakpoint()
                    loss_r = loss_fn(outputs, labels)
                    with torch.no_grad():
                        outputs_baseline = model_downstream(generated_caption_baseline, ques)
                        loss_baseline = loss_fn(outputs_baseline, labels)
                    # else:
                    #     with torch.no_grad():
                    #         outputs = model_downstream(caption, mask)
                    #         loss_r = loss_fn(outputs, labels.float())
                    #         outputs_baseline = model_downstream(caption_baseline, mask_baseline)
                    #         loss_baseline = loss_fn(outputs_baseline, labels.float())
                    # reward = (torch.mean(loss_r,-1) - torch.mean(loss_baseline,-1))
                    reward = loss_r - loss_baseline

                    reward_anchor_r = recap_score_r - base_score_r
                    reward_anchor_c = recap_score_c - base_score_c
                    if recap_score_r <0.1:
                        break
                    # print(reward)
                    # print(reward_anchor)
                    # print(reward)
                    total_reward = reward + 0.5*reward_anchor_r + 0.5*reward_anchor_c
                    loss = -torch.mean(words_prob, -1) * total_reward
                    # breakpoint()
                    # breakpoint()
                    loss = loss.mean()
                    
                    #grad_norm = torch.sqrt(sum([torch.norm(p.data)**2 for p in model_captioning.module.model.qformer.parameters()]))
                    
                    loss.backward()
                    # for p in model_captioning.module.parameters():
                    #     breakpoint()
                    #     param_norm = p.grad.data.norm(2)
                    #     total_norm += param_norm.item() ** 2
                    # total_norm = total_norm ** (1. / 2)
                
                    #grad_norm = torch.sqrt(sum([torch.norm(p.data)**2 for p in model_captioning.module.model.qformer.parameters()]))
                    if it % 3 == 0 or flag ==1:
                        optimizer1.step()
                        optimizer1.zero_grad()
                        torch.cuda.empty_cache()
                    # optimizer1.step()
                    this_loss = loss.item()
                    running_loss1 += this_loss
                    # optimizer1.zero_grad()
                    ####################################################################################################
                    # with torch.no_grad():
                    
                    #     generated_tokens_greedy = model_captioning.module.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=60)
                    #     generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True) 
                        
                    # # outs  = model_captioning.module.model_blip.generate.__wrapped__(model_captioning.module.model_blip, input_ids=prompts, pixel_values=images, do_sample=True, max_length=60,output_logits= True, return_dict_in_generate=True  )
                    # generated_caption_baseline_dict = {key: [value] for key,value in zip(image_id[0], generated_caption_baseline)}
                    # score_after_c = evaluate_cider(generated_caption_baseline_dict, caption_dict)
                    # print("reward = "+str(reward))
                    # print(generated_caption_baseline)
                    # generated_tokens_sampling = outs[0]
                    # logits = outs[1]
                    # logits = torch.stack(logits, dim=1)
                
                    # indices = generated_tokens_sampling.unsqueeze(-1)
                    # words_prob = torch.gather(logits, dim=2, index=indices).squeeze(-1)
                    

                    # # # # print(generated_tokens_sampling)
                    # # generated_caption = processor.batch_decode(generated_tokens_sampling, skip_special_tokens=True) 
                    # generated_tokens_greedy = model_captioning.module.model_blip.generate(input_ids=prompts, pixel_values=images, max_length=60)
                    # generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True) 
                    
                    # generated_caption_dict = {key: [value] for key,value in zip(image_id[0], generated_caption_baseline)}
                    # #caption_dict = {key: [value] for key,value in zip(image_id[0], gt_caption)}



                    # score_after_c = evaluate_cider(generated_caption_dict, caption_dict)
                    # #score_after_s = evaluate_spice(generated_caption_dict, caption_dict)
                    # reward = score_after_c - score_before_c
                    # loss = -torch.mean(words_prob, -1) * reward
                    # loss = loss.mean()
            
                    # loss.backward()

                    # optimizer1.step()

                    if it%25==0:
                        # breakpoint()
                        print(ques)
                        print(generated_caption_baseline_dict_base)
                        print(generated_caption_baseline_dict)
                        
                        # print(loss_r)
                    
                        # s = evaluation(model_captioning, coco_cider_dataloader, ref_caps)
                    if it%750==0:
                            torch.save(model_captioning.module.state_dict(), '/l/users/israfel.salazar/abdo/recaptioning_it_'+str(it)+'.pth')
                    pbar.set_postfix(loss=running_loss1/ (it + 1))
                    pbar.update()
                    loss = running_loss1 / len(things_coco_dataloader_train)
                    wandb.log({"Rouge reward ": reward_anchor_r.item()/2})
                    wandb.log({"down_stream reward  ": reward.mean().item()})
                    wandb.log({"loss  ": loss})
                    wandb.log({"running loss  ": (running_loss1/ (it + 1))})

                #torch.save(model_captioning.module.state_dict(), 'captioning'+str(i)+'.pth')
            # val_loss = validation_down(things_dataloader_val, model)
            # if val_loss < best_val:
            #  best_val = val_loss
            #  torch.save(model.module.state_dict(), 'Downstream_1.pth')
            # print("##########VAL LOSS########: "+ str(val_loss))



def pretraining_down(things_dataloader_train,things_dataloader_val, model, optimizer, weights, scheduler):
    weights = weights.to("cuda")
    #loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = weights**3)

    loss_fn = MultiClassBCELoss()
    # model_downstream_state_dict = torch.load('Downstream_1.pth')
    # model.module.load_state_dict(model_downstream_state_dict)
    model = DDP(model.module)
    model = model.to("cuda")
    best_val = 10000

    for i in range(0,1000):
        running_loss = 0.
        with tqdm( unit='it', total=len(things_dataloader_train)) as pbar:
    
            for  it,(labels, mask, captions) in  enumerate(things_dataloader_train):
            #for  it,(obj, subj, caption, mask, subj_c, obj_c) in  enumerate(things_dataloader_train):                
                labels, captions, mask = labels.to("cuda"), captions.to("cuda"), mask.to("cuda")
                #obj, subj, caption, mask, subj_c, obj_c = obj.to("cuda"), subj.to("cuda"), caption.to("cuda"), mask.to("cuda"), subj_c.to("cuda"), obj_c.to("cuda")
                # breakpoint()
                outputs = model(captions, mask)
                #outputs = model(obj, subj, caption, mask, subj_c, obj_c)
                # breakpoint()
                weights = weights**3
                loss = loss_fn(outputs, labels.float(), weights.repeat(labels.size(0),1))
                #loss = loss_fn(outputs, labels.float())
                loss.backward()
                this_loss = loss.item()
                running_loss += this_loss
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

                loss = running_loss / len(things_dataloader_train)
        val_loss, fscore = validation_down(things_dataloader_val, model)
        scheduler.step(val_loss)

        if val_loss < best_val:
         patience = 0
         best_val = val_loss
         torch.save(model.module.state_dict(), 'Downstream_weighted.pth')
        else:
            patience+=1
        if patience ==50:
            break
        print("##########VAL LOSS########: "+ str(val_loss))
        print("##########F1-Score########: "+ str(fscore))

labels = []
for  it,(label, mask, captions) in  enumerate(things_dataloader_train):
    labels.append(label)
z = sum(labels)
freq = sum(z)
bot = sum(freq)
weights =1-freq/bot
import warnings
warnings.filterwarnings("ignore")
#train_captioning(model_blip, coco_dataloader_train, optimizer1)
#pretraining_down(things_dataloader_train, things_dataloader_val, model_down, optimizer_pretraining, weights, scheduler)
#evaluation(model_blip, coco_dataloader_val, ref_caps)
#recaptioning(things_coco_dataloader_train,things_coco_dataloader_val,coco_dataloader_val, things_dataloader_val, ref_caps, model_blip, model_down, optimizer1, optimizer2, weights )
recaptioning(vqa_train_dataloader,vqa_val_dataloader,coco_dataloader_val, things_dataloader_val, ref_caps, model_blip, model_down, optimizer1, optimizer2, weights )


# model_blip = DDP(model_blip)
# model_blip = model_blip.to("cuda")
# evaluation(model_blip, coco_dataloader_val, ref_caps)



                        # while True:
                        #     outputs = model_captioning(**inputs)
                        #     next_token_logits = outputs.logits[:, -1, :] / temperature
                        #     probs = softmax(next_token_logits, dim=-1)
                        #     next_token = torch.multinomial(probs, num_samples=1)
                        #     output_sequences = torch.cat([output_sequences, next_token], dim=-1)

                        #     inputs["input_ids"] = output_sequences

                        #     if next_token == processor.tokenizer.eos_token_id or output_sequences.shape[1] >= max_length:
                        #         break
                        # # print(generated_tokens_sampling)
                        # generated_caption = processor.batch_decode(generated_tokens_sampling, skip_special_tokens=True) 
                




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
# device = "cuda" if torch.cuda.is_available() else "cpu"
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
