from recaption_model_object import blip_recaption, bert_recaption
from recap_dataset import Coco_Dataset, Coco_Dataset_objects, Coco_Dataset_attributes, Coco_Dataset_Ref
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
# accelerator = Accelerator()
from sklearn.metrics import accuracy_score
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
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
model_down = bert_recaption()

stuff_objects_train = Coco_Dataset_objects()
# stuff_objects_coco_train = Coco_Dataset_objects(downstream= False)#with image
# stuff_objects_coco_val = Coco_Dataset_objects(split="val", downstream= False)#with image
stuff_objects_val = Coco_Dataset_objects(split = "val")


# # Create shuffled indices
# indices = torch.randperm(len(stuff_objects_train)).tolist()

# # Create subsets from the datasets
# coco_train_shuffeled = Subset(coco_train, indices)
# stuff_objects_train_shuffeled = Subset(stuff_objects_train, indices)

batch_size = 32
# model_blip = blip_recaption()
model_down = bert_recaption()

# model_blip = model_blip.to("cuda")
# model_blip = DDP(model_blip)

model_down = model_down.to("cuda")
model_down = DDP(model_down)
for name, param in model_down.module.named_parameters():
        param.requires_grad = True

for name, param in model_down.module.named_parameters():
    if "model" in name or "Bert" in name :
        param.requires_grad = False
# optimizer1 = torch.optim.AdamW(model_blip.parameters(), lr=5e-5)
optimizer2 = torch.optim.SGD(model_down.module.parameters(), lr=0.0001)
# params = list(model_blip.parameters()) + list(model_down.parameters())
optimizer_pretraining = torch.optim.Adam(model_down.module.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_pretraining, 'min', patience = 3, factor = 0.5)
# optimizer3 = torch.optim.Adam(params, lr=0.0001)
device = "cuda" if torch.cuda.is_available() else "cpu"
#coco_dataloader_train = DataLoader(coco_train, batch_size=4,shuffle=True, collate_fn = coco_train.collate_fn, num_workers=1,
#                                    drop_last=True)
# coco_dataloader_val = DataLoader(coco_val, batch_size=32,shuffle=False, collate_fn = coco_val.collate_fn, num_workers=1,
#                                    drop_last=True)

objects_dataloader_train = DataLoader(stuff_objects_train, batch_size=512,shuffle=True, num_workers=4,collate_fn = stuff_objects_train.collate_fn,
                                    drop_last=True) #no image

objects_dataloader_val = DataLoader(stuff_objects_val, batch_size=256,shuffle=True, num_workers=4,collate_fn = stuff_objects_val.collate_fn,
                                    drop_last=True) #no image

# objects_coco_dataloader_train = DataLoader(stuff_objects_coco_train, batch_size=12,shuffle=True, num_workers=4,collate_fn = stuff_objects_coco_train.collate_fn,
#                                     drop_last=True) #with image
# objects_coco_dataloader_val = DataLoader(stuff_objects_coco_val, batch_size=24,shuffle=False, num_workers=4,collate_fn = stuff_objects_coco_val.collate_fn,
#                                     drop_last=True)
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

def evaluate_cider(gen_captions, ref_captions):
    scorer = Cider()
    cider_score, _ = scorer.compute_score(ref_captions, gen_captions)
    print(f"CIDEr Score: {cider_score}")
    return cider_score



ref_caps = load_references_from_json("./my_datasets/annotations/captions_val2017.json")


def evaluation(model, dataloader_val, ref_caps):

    model.eval()
    model = DDP(model.module)
    model = model.to("cuda")
    gen_caps = {}
    with tqdm( unit='it', total=len(dataloader_val)) as pbar:
        for it, (images, captions, mask, prompts, ids) in enumerate(dataloader_val):
            images, captions, prompts = images.to("cuda"), captions.to("cuda"), prompts.to("cuda")

            with torch.no_grad():
                # output = model.module.generate(images,captions,prompts, ids)
                processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
                generated_tokens = model.module.model.generate(input_ids=prompts, pixel_values=images, max_length=60)
                generated_caption = processor.batch_decode(generated_tokens, skip_special_tokens=True)
                output = {key: [value] for key,value in zip(ids[0], generated_caption)}

            gen_caps = {**gen_caps, **output}
            pbar.update()
    # breakpoint()
    ref_caps = dict(list(ref_caps.items())[0:len(gen_caps)])
    score = evaluate_cider(gen_caps, ref_caps)
    return score


def validation_down(things_dataloader_val, model,weights):
        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = torch.nn.BCEWithLogitsLoss()
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = weights**3)
        running_loss = 0.
        y_pred = []
        y_true = []
        y_pred_args = []
        y_true_args = []
        with tqdm( unit='it', total=len(things_dataloader_val)) as pbar:
            threshold = 0.5
            with torch.no_grad():
                for  it,(labels, mask, captions) in  enumerate(things_dataloader_val):
                #for  it,(obj, subj, caption, mask, subj_c, obj_c) in  enumerate(things_dataloader_train):                
                    # labels, captions, mask = labels.to(device), captions.to(device), mask.to(device)
                    labels= labels.to(device)

                    outputs = model(captions, labels.device)
                    #outputs = model(obj, subj, caption, mask, subj_c, obj_c)
                    # breakpoint()
                    loss = loss_fn(outputs, labels)
                    this_loss = loss.item()
                    running_loss += this_loss
                    pbar.set_postfix(loss=running_loss / (it + 1))
                    pbar.update()
                    loss = running_loss / len(things_dataloader_val)
                    outputs_binary = (torch.sigmoid(outputs) > threshold).int()
                    # breakpoint()
                    # preds = outputs.argmax(1)
                    # preds = F.one_hot(out_max, num_classes=200)
                    y_true.extend(labels.int().cpu().tolist())
                    y_pred.extend(outputs_binary.cpu().tolist())
                    # y_true_args.extend(torch.argmax(labels,1).cpu().tolist())
                    # y_pred_args.extend(torch.argmax(outputs,1).cpu().tolist())
                report = classification_report(y_true, y_pred, output_dict=True)
                fscore = report['samples avg']['f1-score']
                #breakpoint()
                #acc2 = accuracy_score(y_true_args, y_pred_args)
               
                #acc = (torch.argmax(torch.FloatTensor(y_pred), 1) == torch.argmax(torch.FloatTensor(y_true), 1)).float().mean()
                
                return loss, fscore

def recaptioning(objects_coco_dataloader_train,coco_dataloader_val, coco_cider_dataloader, objects_dataloader_val, ref_caps, model_captioning, model_downstream, optimizer1, optimizer2):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model_downstream_state_dict = torch.load('Downstream_focal_weighted.pth')
    model_downstream.module.load_state_dict(model_downstream_state_dict)
    for name, param in model_captioning.module.named_parameters():
        param.requires_grad = False
    for name, param in model_captioning.module.model.qformer.named_parameters():
        param.requires_grad = True
    for name, param in model_downstream.module.named_parameters():
        if "model" in name or "Bert" in name :
          param.requires_grad = False
        model_downstream = DDP(model_downstream.module)
    model_downstream = model_downstream.to("cuda")
    model_captioning = DDP(model_captioning.module)
    model_captioning = model_captioning.to("cuda")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    best_val = 10000
    for i in range(0,100):
        running_loss = 0.
        with tqdm( unit='it', total=len(objects_coco_dataloader_train)) as pbar:
            _, f_score = validation_down(objects_dataloader_val, model_downstream)

            s = evaluation(model_captioning, coco_cider_dataloader, ref_caps)

            print("##########CIDEr########: "+ str(s))
            print("##########F1-Score########: "+ str(f_score))

            for  it,(images, captions, mask, prompts, image_id, labels) in  enumerate(objects_coco_dataloader_train):
                images, prompts, labels, captions, mask = images.to("cuda"),prompts.to("cuda"),labels.to("cuda"), captions.to("cuda"), mask.to("cuda")

                with torch.no_grad():
                    generated_tokens_greedy = model_captioning.module.model.generate(input_ids=prompts, pixel_values=images, max_length=60)
                    generated_caption_baseline = processor.batch_decode(generated_tokens_greedy, skip_special_tokens=True) 
            
                if i%2 == 0:
                    outs = model_captioning.module.model.generate(input_ids=prompts, pixel_values=images, max_length=60,num_beams= 3, output_logits= True, return_dict_in_generate=True )
                    generated_tokens_beam = outs[0]
                    logit = outs[1]
                    generated_caption = processor.batch_decode(generated_tokens_beam, skip_special_tokens=True) 
                else:
                    outs  = model_captioning.module.model.generate(input_ids=prompts, pixel_values=images, max_length=60, do_sample = True, output_logits= True, return_dict_in_generate=True  )
                    generated_tokens_sampling = outs[0]
                    logit = outs[1]
                    # print(generated_tokens_sampling)
                    generated_caption = processor.batch_decode(generated_tokens_sampling, skip_special_tokens=True) 
                



                tokenized_caption_baseline = tokenizer(generated_caption_baseline,padding='max_length', truncation=True, max_length=60, return_tensors='pt')
                caption_baseline = tokenized_caption_baseline["input_ids"]
                mask_baseline = tokenized_caption_baseline['attention_mask']
                

                tokenized_caption_ = tokenizer(generated_caption,padding='max_length', truncation=True, max_length=60, return_tensors='pt')
                caption = tokenized_caption_["input_ids"]
                mask = tokenized_caption_['attention_mask']
               
                logits = torch.stack(logit, dim=0)
                logits = torch.mean(logits, dim=0)
                
                outputs = model_downstream(caption, mask)
                loss_r = loss_fn(outputs, labels.float())
                with torch.no_grad():
                    outputs_baseline = model_downstream(caption_baseline, mask_baseline)
                    loss_baseline = loss_fn(outputs_baseline, labels.float())
                # else:
                #     with torch.no_grad():
                #         outputs = model_downstream(caption, mask)
                #         loss_r = loss_fn(outputs, labels.float())
                #         outputs_baseline = model_downstream(caption_baseline, mask_baseline)
                #         loss_baseline = loss_fn(outputs_baseline, labels.float())

                reward = (loss_r - loss_baseline)
                # print(reward)
                loss = -torch.mean(logits, -1) * reward
                loss = loss.mean()
                loss.backward()
                this_loss = loss.item()
                running_loss += this_loss
                optimizer1.step()
                optimizer1.zero_grad()
                if i%2==0:
                    optimizer2.step()
                optimizer2.zero_grad()
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                loss = running_loss / len(objects_coco_dataloader_train)

            #torch.save(model_captioning.module.state_dict(), 'captioning'+str(i)+'.pth')
        # val_loss = validation_down(objects_dataloader_val, model)
        # if val_loss < best_val:
        #  best_val = val_loss
        #  torch.save(model.module.state_dict(), 'Downstream_1.pth')
        # print("##########VAL LOSS########: "+ str(val_loss))



def pretraining_down(objects_dataloader_train,objects_dataloader_val, model, optimizer, weights, scheduler):

    weights = weights.to("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = weights**3)
    patience = 0
    # model_downstream_state_dict = torch.load('Downstream_object.pth')
    # model.module.load_state_dict(model_downstream_state_dict)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # model_downstream_state_dict = torch.load('Downstream_1.pth')
    # model.module.load_state_dict(model_downstream_state_dict)
    model = DDP(model.module)
    model = model.to("cuda")
    best_val = 10000

    for i in range(0,1000):
        running_loss = 0.
        with tqdm( unit='it', total=len(objects_dataloader_train)) as pbar:
    
            for  it,(labels, mask, captions) in  enumerate(objects_dataloader_train):
            #for  it,(obj, subj, caption, mask, subj_c, obj_c) in  enumerate(objects_dataloader_train):                
                # labels, captions, mask = labels.to("cuda"), captions.to("cuda"), mask.to("cuda")
                labels = labels.to("cuda")
                #obj, subj, caption, mask, subj_c, obj_c = obj.to("cuda"), subj.to("cuda"), caption.to("cuda"), mask.to("cuda"), subj_c.to("cuda"), obj_c.to("cuda")
                # breakpoint()
                outputs = model(captions, labels.device)
                #outputs = model(obj, subj, caption, mask, subj_c, obj_c)
               
                loss = loss_fn(outputs, labels)
         
                #loss = loss_fn(outputs, labels.float())
                loss.backward()
                this_loss = loss.item()
                running_loss += this_loss
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

                loss = running_loss / len(objects_dataloader_train)
        val_loss, acc = validation_down(objects_dataloader_val, model, weights)
        scheduler.step(val_loss)

        if val_loss < best_val:
         patience = 0
         best_val = val_loss
         torch.save(model.module.state_dict(), 'Downstream_object_best.pth')
        else:
            patience+=1
        if patience ==80:
            break
        print("##########VAL LOSS########: "+ str(val_loss))
        print("##########f-score########: "+ str(acc))
        torch.save(model.module.state_dict(), 'Downstream_object_last.pth')
        # print("##########accuracy2########: "+ str(acc2))
import numpy as np
labels = []
for  it,(label, mask, captions) in  enumerate(objects_dataloader_train):
    labels.append(label)
data_array = np.array(labels)
# z = sum(labels)
# bot = len(objects_dataloader_train)
batch_class_counts = np.sum(data_array, axis=0)
total_calss_counts = np.sum(batch_class_counts, axis=0)
#total_calss_counts = total_calss_counts + 1e-7
# beta = 0.99
# effective_num = (1 - np.power(beta, total_calss_counts)) / (1 - beta)
# class_weights = 1.0 / effective_num
# class_weights[total_calss_counts == 0] = 0
# class_weights = class_weights / np.sum(class_weights) * len(total_calss_counts)
# class_weights_list = class_weights.tolist()

counts = sum(total_calss_counts)
# bot2 = sum(freq)
weights =1-total_calss_counts/counts
weights = torch.tensor(weights)
import warnings
warnings.filterwarnings("ignore")
# breakpoint()
pretraining_down(objects_dataloader_train, objects_dataloader_val, model_down, optimizer_pretraining, weights, scheduler)
#evaluation(model_blip, coco_dataloader_val, ref_caps)
#recaptioning(objects_coco_dataloader_train,objects_coco_dataloader_val,coco_dataloader_val, objects_dataloader_val, ref_caps, model_blip, model_down, optimizer1, optimizer2 )







# from recaption_model import blip_recaption, bert_recaption
# from recap_dataset import Coco_Dataset, Coco_Dataset_objects
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
# stuff_objects_train = Coco_Dataset_objects()



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

# # objects_dataloader_train = DataLoader(stuff_objects_train, batch_size=512,shuffle=True, num_workers=1,collate_fn = stuff_objects_train.collate_fn,
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
#     stuff_objects_train = Coco_Dataset_objects()
#     sampler = torch.utils.data.distributed.DistributedSampler(stuff_objects_train, num_replicas=world_size, rank=rank)
#     objects_dataloader_train = DataLoader(stuff_objects_train, sampler=sampler, batch_size=512,shuffle=True, num_workers=1,collate_fn = stuff_objects_train.collate_fn,
#                                         drop_last=True)



#     loss_fn = torch.nn.BCEWithLogitsLoss()
#     running_loss = 0.
#     #model = model.to(device)
#     device = torch.device(f'cuda:{rank}')
#     model.to(device)
#     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

#     with tqdm( unit='it', total=len(objects_dataloader_train)) as pbar:
#         for  it,(labels,captions) in  enumerate(objects_dataloader_train):
            
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

#             loss = running_loss / len(objects_dataloader_train)
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