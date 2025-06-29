import torch
from transformers import AutoProcessor
import json
import h5py
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pickle
class Coco_Dataset(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train"):
       self.img_root = "./coco_images.h5"
       if split =="train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
       else:
             self.ann_root = "./my_datasets/annotations/captions_val2017.json"
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
       self.hf = h5py.File(self.img_root, 'r')
       with open(self.ann_root) as f:
            data = json.load(f)
            self.data = data["annotations"]
    #    img_id = self.data[0]["image_id"]
    #    caption = self.data[0]["caption"]
    #    image = self.processor(images=self.hf[str(img_id)][()], return_tensors="pt")
    #    caption = self.processor(text= caption, return_tensors="pt")
   
    #   breakpoint()
    #    x =0


    def __getitem__(self, i):
        img_id = self.data[i]["image_id"]
        caption = self.data[i]["caption"]
        img_jpg = self.hf[str(img_id)][()]

        try:
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
       # caption = self.processor(text= caption, return_tensors="pt")#this need to be in the collate function
        # breakpoint()
        return image,caption, img_id

    def __len__(self):
        return len(self.data)

    
    def collate_fn(self, batch):
   
        images = torch.stack([example[0] for example in batch])
        ids = [[example[2] for example in batch]]
        cap_output = self.processor.tokenizer(text= [example[1] for example in batch], padding='longest', return_tensors="pt")
        #prompt = " "
        # prompt = "a photo of"
        prompt = " "
        
        captions = cap_output["input_ids"]
        prompt_list = [prompt for i in range(0,captions.size()[0])]
        prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]
        mask = cap_output['attention_mask']
        return images, captions, mask, prompt_list, ids


# class stuff_objects(Dataset):
#     def __init__(self, ann_root=None, down_root = None split="train"):
       
#        if split =="train":
#              self.ann_root = "./my_datasets/annotations/captions_train2017.json"
#              self.down_root = "./my_datasets/my_datasets/ordered_stuff_train.json"

#        else:
#              self.ann_root = "./my_datasets/my_datasets/ordered_stuff_val.json"
#              self.down_root = "./my_datasets/my_datasets/ordered_stuff_train.json"
#        with open(self.ann_root) as f:
#             self.data = json.load(f)
            
#     #    img_id = self.data[0]["image_id"]
#     #    caption = self.data[0]["caption"]
#     #    image = self.processor(images=self.hf[str(img_id)][()], return_tensors="pt")
#     #    caption = self.processor(text= caption, return_tensors="pt")
   
#     #   breakpoint()
#     #    x =0


#     def __getitem__(self, i):
        
#         label = self.data[i]["labels"]
#         id = self.data[i]["image_id"]

#         return label, id

#     def __len__(self):
#         return len(self.data)
import numpy as np
class Coco_Dataset_attributes(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "/l/users/israfel.salazar/abdo/coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
       if split =="train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./my_datasets/cocottributes_eccv_version.pkl"
       else:
             self.ann_root = "./my_datasets/annotations/captions_val2017.json"
             self.down_root = "./my_datasets/cocottributes_eccv_version.pkl"
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
       self.hf = h5py.File(self.img_root, 'r')
       with open(self.ann_root) as f:
            data = json.load(f)
            self.data = data["annotations"]
       with open(self.down_root, 'rb') as file:
            self.down = pickle.load(file, encoding='latin1')
    #    img_id = self.data[1]["image_id"]
    #    caption = self.data[0]["caption"]
    #    image = self.processor(images=self.hf[str(img_id)][()], return_tensors="pt")
    #    caption = self.processor(text= caption, return_tensors="pt")
    #    breakpoint()
    #    test = self.down['ann_vecs'][img_id]
       
    #    x =0


    def __getitem__(self, i):
        img_id = self.data[i]["image_id"]
        caption = self.data[i]["caption"]
        
        if self.downstream:
            try:
                label = self.down['ann_vecs'][img_id]
            except:    
                # print("missing")
                label = np.zeros(204)
            return label,caption


        img_jpg = self.hf[str(img_id)][()]

        try:
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
       # caption = self.processor(text= caption, return_tensors="pt")#this need to be in the collate function
        # breakpoint()
        return image,caption, img_id

    def __len__(self):
        return len(self.data)

    
    def collate_fn(self, batch):
        # if self.downstream:
        #     label = [example[0] for example in batch]
        #     label = torch.tensor(label)
        #     caption = [example[1] for example in batch]
        label = []
        caption = []
        for l,c in batch:
            if sum(l) == 0:
                continue
            label.append(l)
            caption.append(c)
        label = torch.tensor(label)
        #tokenized = self.tokenizer(caption,padding=True, return_tensors='pt')
        tokenized = self.tokenizer(caption,padding='longest', return_tensors='pt')
        caption = tokenized["input_ids"]
        mask = tokenized['attention_mask']
        if self.downstream == True:
         return label, mask, caption
        images = torch.stack([example[0] for example in batch])
        ids = [[example[2] for example in batch]]
        cap_output = self.processor.tokenizer(text= [example[1] for example in batch], padding='longest', return_tensors="pt")
        prompt = "a photo of"
        #prompt = "descripe the image"
        
        captions = cap_output["input_ids"]
        prompt_list = [prompt for i in range(0,captions.size()[0])]
        prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]
        mask = cap_output['attention_mask']
        return images, captions, mask, prompt_list, ids
import time
import pandas as pd
class Coco_Dataset_things(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "./coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
       if split =="train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./my_datasets/stuff_objects_multilabel_train.json"
       else:
             self.ann_root = "./my_datasets/annotations/captions_val2017.json"
             self.down_root = "./my_datasets/stuff_objects_multilabel_val.json"
       self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
       self.hf = h5py.File(self.img_root, 'r')
       self.down = pd.read_json(self.down_root)
       with open(self.ann_root) as f:
            data = json.load(f)
            self.data = data["annotations"]
            self.data = pd.DataFrame(self.data)
    #    with open(self.down_root) as f:
    #         self.down = json.load(f)
    #    breakpoint()
       
    #    img_id = self.data[0]["image_id"]
    #    caption = self.data[0]["caption"]
    #    image = self.processor(images=self.hf[str(img_id)][()], return_tensors="pt")
    #    caption = self.processor(text= caption, return_tensors="pt")
    #    label =  self.down[self.down["image_id"]==img_id]["labels"]
    #    breakpoint()
    #    x =0


    def __getitem__(self, i):
        # img_id = self.data[i]["image_id"]
        # caption = self.data[i]["caption"]
        img_id = self.down.iloc[i]["image_id"]
        captions = self.data[self.data["image_id"]==img_id]
        caption = captions.iloc[random.randint(len(captions)-1)]["caption"]
        label = self.down.iloc[i]["labels"]

        
        # try:
        #     label =  self.down[self.down["image_id"]==img_id]["labels"].iloc[0]
        # except Exception as f:
        #     # print(f)
        #     label = np.zeros(182)
        # label_id = self.down[i]["image_id"]
        # if img_id != label_id:
        #     for labels in self.down:
        #         if labels["image_id"] == img_id:
        #             label = labels["labels"]
        if self.downstream:
            return label,caption
        
        img_jpg = self.hf[str(img_id)][()]

        try:
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
       # caption = self.processor(text= caption, return_tensors="pt")#this need to be in the collate function
        # breakpoint()
        return image,caption, label, img_id

    def __len__(self):
        return len(self.down)

    
    def collate_fn(self, batch):
        if self.downstream:
            label = [example[0] for example in batch]
            label = torch.tensor(label)
            caption = [example[1] for example in batch]
            #tokenized = self.tokenizer(caption,padding=True, return_tensors='pt')
            tokenized = self.tokenizer(caption,padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            caption = tokenized["input_ids"]
            mask = tokenized['attention_mask']
            return label, mask, caption
        images = torch.stack([example[0] for example in batch])
        cap_output = self.processor.tokenizer(text= [example[1] for example in batch], padding='longest', return_tensors="pt")
        label = [example[2] for example in batch]
        label = torch.tensor(label)
        ids = [[example[3] for example in batch]]
        prompt = " "
        #prompt = "descripe the image"
        captions = cap_output["input_ids"]
        prompt_list = [prompt for i in range(0,captions.size()[0])]
        prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]
        mask = cap_output['attention_mask']

        return images, captions, mask, prompt_list, ids, label
    
class Coco_Dataset_objects(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "/l/users/israfel.salazar/abdo/coco_images.h5"
       self.downstream = downstream
       #self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
       if split =="train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "/home/israfel.salazar/abdo/Recaption/objects_train_multi_augmented.json"
       else:
             self.ann_root = "./my_datasets/annotations/captions_val2017.json"
             self.down_root = "/home/israfel.salazar/abdo/Recaption/objects_val_multi_augmented.json"
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
       self.hf = h5py.File(self.img_root, 'r')
       self.down = pd.read_json(self.down_root)
       with open(self.ann_root) as f:
            data = json.load(f)
            self.data = data["annotations"]
            self.data = pd.DataFrame(self.data)
    #    with open(self.down_root) as f:
    #         self.down = json.load(f)
    #    breakpoint()
       
    #    img_id = self.data[0]["image_id"]
    #    caption = self.data[0]["caption"]
    #    image = self.processor(images=self.hf[str(img_id)][()], return_tensors="pt")
    #    caption = self.processor(text= caption, return_tensors="pt")
    #    label =  self.down[self.down["image_id"]==img_id]["labels"]
    #    record = self.down.iloc[0]
    #    breakpoint()
    #    x =0


    def __getitem__(self, i):
        # img_id = self.data[i]["image_id"]
        # caption = self.data[i]["caption"]
        img_id = self.down.iloc[i]["image_id"]
        # captions = self.data[self.data["image_id"]==img_id]
        caption = self.down.iloc[i]["caption"]
        label = self.down.iloc[i]["object"]
        
        
        # try:
        #     label =  self.down[self.down["image_id"]==img_id]["labels"].iloc[0]
        # except Exception as f:
        #     # print(f)
        #     label = np.zeros(182)
        # label_id = self.down[i]["image_id"]
        # if img_id != label_id:
        #     for labels in self.down:
        #         if labels["image_id"] == img_id:
        #             label = labels["labels"]
        if self.downstream:
            return label,caption
        
        img_jpg = self.hf[str(img_id)][()]

        try:
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
       # caption = self.processor(text= caption, return_tensors="pt")#this need to be in the collate function
        # breakpoint()
        return image,caption, label, img_id

    def __len__(self):
        return len(self.down)

    
    def collate_fn(self, batch):
        if self.downstream:
            label = [example[0] for example in batch]
            label = torch.tensor(label)
            caption = [example[1] for example in batch]
            #tokenized = self.tokenizer(caption,padding=True, return_tensors='pt')
            # tokenized = self.tokenizer(caption,padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            # caption = tokenized["input_ids"]
            # mask = tokenized['attention_mask']
            return label, None, caption
        images = torch.stack([example[0] for example in batch])
        cap_output = self.processor.tokenizer(text= [example[1] for example in batch], padding='longest', return_tensors="pt")
        label = [example[2] for example in batch]
        label = torch.tensor(label)
        ids = [[example[3] for example in batch]]
        prompt = "a photo of"
        #prompt = "descripe the image"
        captions = cap_output["input_ids"]
        prompt_list = [prompt for i in range(0,captions.size()[0])]
        prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]
        mask = cap_output['attention_mask']

        return images, captions, mask, prompt_list, ids, label
import pandas as pd
from numpy import random
class Coco_Dataset_Ref(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "/l/users/israfel.salazar/abdo/coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
       if split =="train":
             self.ann_root = "./annotations/captions_train2014.json"
             self.down = pd.read_csv("./my_datasets/Training_data_ref-coco_concept.csv")
    #    else:
    #          self.ann_root = "./my_datasets/annotations/captions_val2017.json"
    #          self.down_root = "./my_datasets/my_datasets/ordered_stuff_val.json"
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
       self.hf = h5py.File(self.img_root, 'r')
       with open(self.ann_root) as f:
            data = json.load(f)
            self.data = data["annotations"]
       #breakpoint()


    def __getitem__(self, i):
        img_id = self.data[i]["image_id"]
        caption = self.data[i]["caption"]
        if self.downstream:
            samples = self.down[self.down["img_idx"] == img_id]
            try:
             sample = samples[random.randint(len(samples)-1)] #since their are multiple captions, we pick on relation in random
            except:
                pass
            
            verb = sample["rel"]
            objects = sample["obj"]
            subject = sample["subj"]
            person_x = sample["subj_ctr_x"]
            person_y = sample["subj_ctr_y"]
            person_std_x = sample["subj_sd_x"]
            person_std_y = sample["subj_sd_y"]
            object_x = sample["obj_ctr_x"]
            object_y = sample["obj_ctr_y"]
            object_std_x = sample["obj_sd_x"]
            object_std_y = sample["obj_sd_y"]
            return objects, subj, caption, [person_x,person_y, person_std_x, person_std_y], [object_x, object_y, object_std_x, object_std_y]
        
        img_jpg = self.hf[str(img_id)][()]

        try:
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
       # caption = self.processor(text= caption, return_tensors="pt")#this need to be in the collate function
        # breakpoint()
        return image,caption, img_id

    def __len__(self):
        return len(self.down)

    
    def collate_fn(self, batch):
        if self.downstream:
            obj = [example[0] for example in batch]
            tokenized = self.tokenizer(obj,padding=True, return_tensors='pt')
            obj = tokenized["input_ids"]
            subj = [example[1] for example in batch]
            tokenized = self.tokenizer(subj,padding=True, return_tensors='pt')
            subj = tokenized["input_ids"]
            caption = [example[2] for example in batch]
            #tokenized = self.tokenizer(caption,padding=True, return_tensors='pt')
            tokenized = self.tokenizer(caption,padding='max_length', truncation=True, max_length=60, return_tensors='pt')
            caption = tokenized["input_ids"]
            mask = tokenized['attention_mask']
            subj_c = [example[3] for example in batch]
            subj_c = torch.tensor(subj_c)
            obj_c = [example[4] for example in batch]
            obj_c = torch.tensor(obj_c)

            return obj, subj, caption, mask, subj_c, obj_c
        images = torch.stack([example[0] for example in batch])
        ids = [[example[2] for example in batch]]
        cap_output = self.processor.tokenizer(text= [example[1] for example in batch], padding='longest', return_tensors="pt")
        prompt = "a photo of"
        #prompt = "descripe the image"
        
        captions = cap_output["input_ids"]
        prompt_list = [prompt for i in range(0,captions.size()[0])]
        prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]
        mask = cap_output['attention_mask']
        return images, captions, mask, prompt_list, ids
        
# class Coco_Dataset_VQA(Dataset):
#     def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
#        self.img_root = "/l/users/israfel.salazar/abdo/coco_images.h5"
#        self.downstream = downstream
#        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

#        if split =="train":
#              self.ann_root = "./my_datasets/annotations/captions_train2017.json"
#              self.down_root = "/home/israfel.salazar/abdo/Recaption/really_what_only_100_train_binary_vector_augmented.json"
#        else:
#              self.ann_root = "/home/israfel.salazar/abdo/Recaption/my_datasets/captions_val2014.json"
#              self.down_root = "/home/israfel.salazar/abdo/Recaption/really_what_only_100_binary_val_vector_augmented.json"
#        self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
#     #    self.processor2 = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
#        self.hf = h5py.File(self.img_root, 'r')
 
#        self.down = pd.read_json(self.down_root)
#        with open(self.ann_root) as f:
#             data = json.load(f)
#             self.data = data["annotations"]
#             self.data = pd.DataFrame(self.data)
#     #    with open(self.down_root) as f:
#     #         self.down = json.load(f)
#     #    breakpoint()
       
#     #    img_id = self.data.iloc[5]["image_id"]
#     #    caption = self.data[0]["caption"]
#     #    image = self.processor(images=self.hf[str(img_id)][()], return_tensors="pt")
#     #    caption = self.processor(text= caption, return_tensors="pt")
#     #    breakpoint()
#     #    self.down["answer_vector"].value_counts()
#     #    x = 0
#     #    label =  self.down[self.down["image_id"]==img_id]["labels"]


#     def __getitem__(self, i):
#         img_id = self.down.iloc[i]["image_id"]
#         # captions = self.data[self.data["image_id"]==img_id]
#         # caption = captions.iloc[random.randint(len(captions)-1)]["caption"]
#         caption = self.down.iloc[i]["caption"]
#         label = self.down.iloc[i]["answer_vector"]
#         ques = self.down.iloc[i]["question"]
#         # label_id = self.down[i]["image_id"]
#         # if img_id != label_id:
#         #     for labels in self.down:
#         #         if labels["image_id"] == img_id:
#         #             label = labels["labels"]
#         if self.downstream:
#             return label,caption, ques
        
#         img_jpg = self.hf[str(img_id)][()]

#         try:
#             image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
#         except: # A WILD GREY IMAGE APPEARS!!
#             rgb_image = np.expand_dims(img_jpg, axis=-1)
#             rgb_image = np.repeat(rgb_image, 3, axis=-1)
#             image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
#        # caption = self.processor(text= caption, return_tensors="pt")#this need to be in the collate function
#         # breakpoint()
#         return image,caption, label, ques, img_id

#     def __len__(self):
#         return len(self.down)

    
#     def collate_fn(self, batch):
#         if self.downstream:
#             label = [example[0] for example in batch]
#             label = torch.tensor(label)
#             caption = [example[1] for example in batch]
#             ques = [example[2] for example in batch]
#             # tokenized = self.tokenizer([c + ' ' + q for c, q in zip(caption, ques)], padding='longest', return_tensors='pt')
#             # inputs = tokenized["input_ids"]
#             # mask = tokenized['attention_mask']
#             return label, caption, ques
#         images = torch.stack([example[0] for example in batch])
#         label = [example[2] for example in batch]
#         label = torch.tensor(label)
#         caption = [example[1] for example in batch]
#         ques = [example[3] for example in batch]
#         #ques = torch.tensor(ques)
#        # tokenized = self.tokenizer([c + ' ' + q for c, q in zip(caption, ques)], padding='longest', return_tensors='pt')
# #        inputs = tokenized["input_ids"]
#  #       mask = tokenized['attention_mask']
#         ids = [[example[4] for example in batch]]
#         prompt = "a photo of"
#         #prompt = "descripe the image"
#         # captions = cap_output["input_ids"]
#         prompt_list = [prompt for i in range(0,len(ques))]
#         prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]
#         # mask = cap_output['attention_mask']

#         return images, ques, None, prompt_list, ids, label, caption
    



class Coco_Dataset_dummy(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "./coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

       if split =="train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "/home/israfel.salazar/abdo/Recaption/DQA/new_dummy_train.json"
       else:
             self.ann_root = "/home/israfel.salazar/abdo/Recaption/my_datasets/captions_val2014.json"
             self.down_root = "/home/israfel.salazar/abdo/Recaption/DQA/new_dummy_val.json"
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
       self.hf = h5py.File(self.img_root, 'r')
       self.down = pd.read_json(self.down_root)
       with open(self.ann_root) as f:
            data = json.load(f)
            self.data = data["annotations"]
            self.data = pd.DataFrame(self.data)
    #    with open(self.down_root) as f:
    #         self.down = json.load(f)
    #    breakpoint()
       
    #    img_id = self.data[5]["image_id"]
    #    caption = self.data[0]["caption"]
    #    image = self.processor(images=self.hf[str(img_id)][()], return_tensors="pt")
    #    caption = self.processor(text= caption, return_tensors="pt")
    #    breakpoint()
    #    label =  self.down[self.down["image_id"]==img_id]["labels"]
       
    #    x =0


    def __getitem__(self, i):
        img_id = self.down.iloc[i]["image_id"]
        # captions = self.data[self.data["image_id"]==img_id]
        caption = self.down.iloc[i]["caption"]
        label = self.down.iloc[i]["answer_vector"]
        ques = self.down.iloc[i]["question"]
        # label_id = self.down[i]["image_id"]
        # if img_id != label_id:
        #     for labels in self.down:
        #         if labels["image_id"] == img_id:
        #             label = labels["labels"]
        if self.downstream:
            return label,caption, ques
        
        img_jpg = self.hf[str(img_id)][()]

        try:
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        
       # caption = self.processor(text= caption, return_tensors="pt")#this need to be in the collate function
        # breakpoint()
        return image,caption, label, img_id

    def __len__(self):
        return len(self.down)

    
    def collate_fn(self, batch):
        if self.downstream:

            label = [example[0] for example in batch]
            label = torch.tensor(label)
            caption = [example[1] for example in batch]
            ques = [example[2] for example in batch]
            tokenized = self.tokenizer([c + ' ' + q for c, q in zip(caption, ques)], padding='longest', return_tensors='pt')
            inputs = tokenized["input_ids"]
            mask = tokenized['attention_mask']
            # label = [example[0] for example in batch]
            # label = torch.tensor(label)
            # caption = [example[1] for example in batch]
            # ques = [example[2] for example in batch]

            # tokenized = self.tokenizer(caption,padding=True, return_tensors='pt')
            # tokenized = self.tokenizer(caption, padding='longest', return_tensors='pt')
            # caption = tokenized["input_ids"]
            # mask_cap = tokenized['attention_mask']
            # tokenized = self.tokenizer(ques, padding='longest', return_tensors='pt')
            # ques = tokenized["input_ids"]
            # mask_ques = tokenized['attention_mask']
            # inputs = torch.cat((ques,caption), 1)
            # mask = torch.cat((mask_ques,mask_cap), 1)
            # # inputs = [q+c for q,c in zip(ques,caption)]
            # # tokenized = self.tokenizer(inputs, padding='longest', return_tensors='pt')
            # # inputs = tokenized["input_ids"]
            # # mask = tokenized['attention_mask']
            return label, inputs, mask
        images = torch.stack([example[0] for example in batch])
        cap_output = self.processor.tokenizer(text= [example[1] for example in batch], padding='longest', return_tensors="pt")
        label = [example[2] for example in batch]
        label = torch.tensor(label)
        ids = [[example[3] for example in batch]]
        prompt = "a photo of"
        #prompt = "descripe the image"
        captions = cap_output["input_ids"]
        prompt_list = [prompt for i in range(0,captions.size()[0])]
        prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]
        mask = cap_output['attention_mask']

        return images, captions, mask, prompt_list, ids, label













from transformers import AutoTokenizer, RobertaModel
from huggingface_hub import login

class Coco_Dataset_VQA(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       hug_token = "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"
       login(token = hug_token)
       self.img_root = "./coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

       if split == "train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
            #  self.down = self.down.sample(n=10000, random_state=42)

       elif split == "finetune":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=50000)

       else:
             self.ann_root = "./my_datasets/captions_val2014.json"
             self.down_root = "./VQA_datasets/VQA_BIO_val.json"
             self.down = pd.read_json(self.down_root)
            #  self.down = self.down.sample(n=20000)

    #    self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
       self.hf = h5py.File(self.img_root, 'r')
 
       with open(self.ann_root) as f:
            data = json.load(f)
            
            self.data = data["annotations"]
            self.data = pd.DataFrame(self.data)
            self.data = self.data.drop_duplicates(subset="image_id", keep="first")

            if split =="val":
                self.data = self.data.sample(n=300,  random_state=42)

    #    img_id = self.down.iloc[0]["image_id"]
    #    prompt = "caption en"
    #    z = self.processor(text=prompt, images=img_jpg, return_tensors="pt")
    #    breakpoint()
    #    caption = self.down.iloc[0]["caption"]
    #    span_start = self.down.iloc[0]["start_span"]
    #    span_end = self.down.iloc[0]["end_span"]
    #    ques = self.down.iloc[0]["question"]
    #    answer = self.down.iloc[0]["answer"]
    #    tokenized = self.tokenizer(caption, ques, padding='longest', return_tensors="pt",  add_special_tokens=True)["input_ids"]
      
    #    breakpoint()

    def __getitem__(self, i):
        img_id = self.data.iloc[i]["image_id"]

        caption = self.data.iloc[i]["caption"].lower()
        # span_start = self.down.iloc[i]["start_span"]
        # span_end = self.down.iloc[i]["end_span"]
        label = self.down.iloc[0]["label"]
        ques = self.down.iloc[0]["question"]
        answer = self.down.iloc[0]["answer"]
        if self.downstream:
            return label,caption, ques, answer
        
        img_jpg = self.hf[str(img_id)][()]
        prompt = "caption en"

        try:
            input_ids, _, image = self.processor(text=prompt, images=img_jpg, return_tensors="pt").values()
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            input_ids, _, image = self.processor(text=prompt, images=rgb_image, return_tensors="pt").values()
        

        return label,caption, ques, answer, image.squeeze(0), img_id, input_ids.squeeze(0)


    def __len__(self):
        return len(self.data)

    
    
    def collate_fn(self, batch):
        
        label = [example[0] for example in batch]
        
        max_len = max(len(size) for size in label)
        label = [example + [0] * (max_len - len(example)) for example in label] #pad the labels to equal size, since they depend on no. tokens
        label = torch.tensor(label)
        caption = [example[1] for example in batch]
        ques = [example[2] for example in batch]
        answer = [example[3] for example in batch]
            # ques = [que + self.tokenizer.sep_token for que in ques]
        # ques = [que + self.tokenizer.sep_token for que in ques]

        # sep_token_id = self.tokenizer.sep_token_id
        # sep_token_position = (tokenized_inputs == sep_token_id).nonzero(as_tuple=True)[1][0].item()
        # attn_mask[sep_token_position::] = 0
        # attn_mask = torch.tensor(attn_mask)
        if self.downstream:
            tokenized = self.tokenizer(ques, caption, padding='longest', return_tensors="pt")
            
            tokenized_input = tokenized["input_ids"]
#pad the labels to be equal to inputs padding
            padding_size = tokenized_input.size(1) - label.size(1)
            label = torch.nn.functional.pad(label, (0, padding_size)) 
            # hot_label = torch.nn.functional.one_hot(label, num_classes=3)
            attn_mask = tokenized["attention_mask"]
            return label, tokenized_input, attn_mask, answer, caption, ques
        images = torch.stack([example[4] for example in batch])
        ids = [[example[5] for example in batch]]
        # prompt = "caption en"
        input_ids = torch.stack([example[6] for example in batch])
        # prompt_list = [prompt for i in range(0,len(ques))]
        # prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]


        return label, answer, caption, ques, images, input_ids, ids


    
class Coco_Dataset_VQA_Comparision(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "./coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

       if split == "train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=1000, random_state=42)

       elif split == "finetune":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=50000)

       else:
             self.ann_root = "./my_datasets/captions_val2014.json"
             self.down_root = "./VQA_datasets/VQA_BIO_val.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=1000, random_state=42)

       self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
       self.processor2 = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
       self.hf = h5py.File(self.img_root, 'r')
 
       with open(self.ann_root) as f:
            data = json.load(f)
            
            self.data = data["annotations"]
            self.data = pd.DataFrame(self.data)
    #    img_id = self.down.iloc[0]["image_id"]
    #    breakpoint()
    #    caption = self.down.iloc[0]["caption"]
    #    span_start = self.down.iloc[0]["start_span"]
    #    span_end = self.down.iloc[0]["end_span"]
    #    ques = self.down.iloc[0]["question"]
    #    answer = self.down.iloc[0]["answer"]
    #    tokenized = self.tokenizer(caption, ques, padding='longest', return_tensors="pt",  add_special_tokens=True)["input_ids"]
      
    #    breakpoint()


    def __getitem__(self, i):
        img_id = self.down.iloc[i]["image_id"]

        caption = self.down.iloc[i]["caption"].lower()
        # span_start = self.down.iloc[i]["start_span"]
        # span_end = self.down.iloc[i]["end_span"]
        label = self.down.iloc[i]["label"]
        ques = self.down.iloc[i]["question"]
        answer = self.down.iloc[i]["answer"]
        if self.downstream:
            return label,caption, ques, answer
        
        img_jpg = self.hf[str(img_id)][()]
        
        try:
            clip_img = torch.tensor(img_jpg).permute(2,0,1)
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
            image2 = self.processor2(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)

        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            clip_img = torch.tensor(rgb_image).permute(2,0,1)

            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
            image2 = self.processor2(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)

        

        return label,caption, ques, answer, image, img_id, image2, clip_img


    def __len__(self):
        return len(self.down)

    
    
    def collate_fn(self, batch):
        
        label = [example[0] for example in batch]
        
        max_len = max(len(size) for size in label)
        label = [example + [0] * (max_len - len(example)) for example in label] #pad the labels to equal size, since they depend on no. tokens
        label = torch.tensor(label)
        caption = [example[1] for example in batch]
        ques = [example[2] for example in batch]
        answer = [example[3] for example in batch]
            # ques = [que + self.tokenizer.sep_token for que in ques]
        # ques = [que + self.tokenizer.sep_token for que in ques]

        # sep_token_id = self.tokenizer.sep_token_id
        # sep_token_position = (tokenized_inputs == sep_token_id).nonzero(as_tuple=True)[1][0].item()
        # attn_mask[sep_token_position::] = 0
        # attn_mask = torch.tensor(attn_mask)
        if self.downstream:
            tokenized = self.tokenizer(ques, caption, padding='longest', return_tensors="pt")
            
            tokenized_input = tokenized["input_ids"]
#pad the labels to be equal to inputs padding
            padding_size = tokenized_input.size(1) - label.size(1)
            label = torch.nn.functional.pad(label, (0, padding_size)) 
            # hot_label = torch.nn.functional.one_hot(label, num_classes=3)
            attn_mask = tokenized["attention_mask"]
            return label, tokenized_input, attn_mask, answer, caption, ques
        images = torch.stack([example[4] for example in batch])
        images2 = torch.stack([example[6] for example in batch])
        ids = [[example[5] for example in batch]]
        prompt = "a photo of"

        prompt_lists = [prompt for i in range(0,len(ques))]
        prompt_list = self.processor.tokenizer(text= [example for example in prompt_lists], padding='longest', return_tensors="pt")["input_ids"]
        prompt_list2 = self.processor2.tokenizer(text= [example for example in prompt_lists], padding='longest', return_tensors="pt")["input_ids"]
        cilp = [example[7] for example in batch]


        return label, answer, caption, ques, images, prompt_list, ids, images2, prompt_list2, cilp

from PIL import Image
import io

class Pokemon_Dataset_VQA(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "./coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

       if split == "train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=10000, random_state=42)

       elif split == "finetune":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=50000)

       else:
             self.ann_root = "./my_datasets/captions_val2014.json"
             self.down_root = "./VQA_datasets/VQA_BIO_val.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=20000)

    #    self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
       self.processor = AutoProcessor.from_pretrained("gauthampughazhendhi/blip-2-pokemon-captions-fine-tuned")
       self.hf = h5py.File(self.img_root, 'r')
 

       self.data = pd.read_parquet("hf://datasets/tungdop2/pokemon/data/train-00000-of-00001-5e017125a702cfbb.parquet")
    #    img_id = self.down.iloc[0]["image_id"]
    #    caption = self.down.iloc[0]["caption"]
    #    span_start = self.down.iloc[0]["start_span"]
    #    span_end = self.down.iloc[0]["end_span"]
    #    ques = self.down.iloc[0]["question"]
    #    answer = self.down.iloc[0]["answer"]
    #    tokenized = self.tokenizer(caption, ques, padding='longest', return_tensors="pt",  add_special_tokens=True)["input_ids"]
      
    #    breakpoint()

    def __getitem__(self, i):
        # img_id = self.down.iloc[i]["image_id"]

        caption = self.data.iloc[i]["caption"].lower()
        # span_start = self.down.iloc[i]["start_span"]
        # span_end = self.down.iloc[i]["end_span"]
        label = self.down.iloc[i]["label"]
        ques = self.down.iloc[i]["question"]
        answer = self.down.iloc[i]["answer"]
        if self.downstream:
            return label,caption, ques, answer
        img_dict = self.data.iloc[i]['image']
        img_data = img_dict["bytes"]
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_jpg = np.asarray(image)
        # img_jpg = self.hf[str(img_id)][()]

        try:
            image = self.processor(images=img_jpg, return_tensors="pt")["pixel_values"].squeeze(0)
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            image = self.processor(images=rgb_image, return_tensors="pt")["pixel_values"].squeeze(0)
        

        return label,caption, ques, answer, image, None


    def __len__(self):
        return len(self.data)

    
    
    def collate_fn(self, batch):
        
        label = [example[0] for example in batch]
        
        max_len = max(len(size) for size in label)
        label = [example + [0] * (max_len - len(example)) for example in label] #pad the labels to equal size, since they depend on no. tokens
        label = torch.tensor(label)
        caption = [example[1] for example in batch]
        ques = [example[2] for example in batch]
        answer = [example[3] for example in batch]
            # ques = [que + self.tokenizer.sep_token for que in ques]
        # ques = [que + self.tokenizer.sep_token for que in ques]

        # sep_token_id = self.tokenizer.sep_token_id
        # sep_token_position = (tokenized_inputs == sep_token_id).nonzero(as_tuple=True)[1][0].item()
        # attn_mask[sep_token_position::] = 0
        # attn_mask = torch.tensor(attn_mask)
        if self.downstream:
            tokenized = self.tokenizer(ques, caption, padding='longest', return_tensors="pt")
            
            tokenized_input = tokenized["input_ids"]
#pad the labels to be equal to inputs padding
            padding_size = tokenized_input.size(1) - label.size(1)
            label = torch.nn.functional.pad(label, (0, padding_size)) 
            # hot_label = torch.nn.functional.one_hot(label, num_classes=3)
            attn_mask = tokenized["attention_mask"]
            return label, tokenized_input, attn_mask, answer, caption, ques
        images = torch.stack([example[4] for example in batch])
        ids = [[example[5] for example in batch]]
        prompt = ""

        prompt_list = [prompt for i in range(0,len(ques))]
        prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]


        return label, answer, caption, ques, images, prompt_list, ids
import os
class FineCapEval(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "./coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

       if split == "train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=10000, random_state=42)

       elif split == "finetune":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=50000)

       else:
             self.ann_root = "./my_datasets/captions_val2014.json"
             self.down_root = "./VQA_datasets/VQA_BIO_val.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=20000)

    #    self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")

       
       self.hf = h5py.File(self.img_root, 'r')
       self.images_path = "./FineCapEval/images"
 

       self.data = pd.read_csv("./FineCapEval/FineCapEval.csv")
    #    breakpoint()
    #    img_id = self.data.iloc[0]["image"]
    #    caption = self.data.iloc[0]["caption"]
    #    breakpoint()
    #    span_start = self.down.iloc[0]["start_span"]
    #    span_end = self.down.iloc[0]["end_span"]
    #    ques = self.down.iloc[0]["question"]
    #    answer = self.down.iloc[0]["answer"]
    #    tokenized = self.tokenizer(caption, ques, padding='longest', return_tensors="pt",  add_special_tokens=True)["input_ids"]
      
    #    breakpoint()

    def __getitem__(self, i):
        img_id = self.data.iloc[i]["image"]

        caption = self.data.iloc[i]["caption"].lower()
        # span_start = self.down.iloc[i]["start_span"]
        # span_end = self.down.iloc[i]["end_span"]
        label = self.down.iloc[i]["label"]
        ques = self.down.iloc[i]["question"]
        answer = self.down.iloc[i]["answer"]
        if self.downstream:
            return label,caption, ques, answer
        path = os.path.join(self.images_path,img_id)

        image = Image.open(path).convert("RGB")
        img_jpg = np.asarray(image)
        # img_jpg = self.hf[str(img_id)][()]

        prompt = "caption en"

        try:
            input_ids, _, image = self.processor(text=prompt, images=img_jpg, return_tensors="pt").values()
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            input_ids, _, image = self.processor(text=prompt, images=rgb_image, return_tensors="pt").values()
        

        return label,caption, ques, answer, image.squeeze(0), img_id, input_ids.squeeze(0)


    def __len__(self):
        return len(self.data)

    
    
    def collate_fn(self, batch):
        
        label = [example[0] for example in batch]
        
        max_len = max(len(size) for size in label)
        label = [example + [0] * (max_len - len(example)) for example in label] #pad the labels to equal size, since they depend on no. tokens
        label = torch.tensor(label)
        caption = [example[1] for example in batch]
        ques = [example[2] for example in batch]
        answer = [example[3] for example in batch]
            # ques = [que + self.tokenizer.sep_token for que in ques]
        # ques = [que + self.tokenizer.sep_token for que in ques]

        # sep_token_id = self.tokenizer.sep_token_id
        # sep_token_position = (tokenized_inputs == sep_token_id).nonzero(as_tuple=True)[1][0].item()
        # attn_mask[sep_token_position::] = 0
        # attn_mask = torch.tensor(attn_mask)
        if self.downstream:
            tokenized = self.tokenizer(ques, caption, padding='longest', return_tensors="pt")
            
            tokenized_input = tokenized["input_ids"]
#pad the labels to be equal to inputs padding
            padding_size = tokenized_input.size(1) - label.size(1)
            label = torch.nn.functional.pad(label, (0, padding_size)) 
            # hot_label = torch.nn.functional.one_hot(label, num_classes=3)
            attn_mask = tokenized["attention_mask"]
            return label, tokenized_input, attn_mask, answer, caption, ques
        images = torch.stack([example[4] for example in batch])
        ids = [[example[5] for example in batch]]
        # prompt = "caption en"
        input_ids = torch.stack([example[6] for example in batch])
        # prompt_list = [prompt for i in range(0,len(ques))]
        # prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]


        return label, answer, caption, ques, images, input_ids, ids



class DOCCI(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "./coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

       if split == "train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=10000, random_state=42)

       elif split == "finetune":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=50000)

       else:
             self.ann_root = "./my_datasets/captions_val2014.json"
             self.down_root = "./VQA_datasets/VQA_BIO_val.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=20000)

    #    self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
    #    self.hf = h5py.File(self.img_root, 'r')
       self.images_path = "./docci/images"
 

       self.data = pd.read_json("./docci/docci_descriptions.jsonlines", lines=True)
       self.data = self.data[self.data["split"]=="test"]
    #    breakpoint()
    #    img_id = self.data.iloc[0]["image"]
    #    caption = self.data.iloc[0]["caption"]
    #    breakpoint()
    #    span_start = self.down.iloc[0]["start_span"]
    #    span_end = self.down.iloc[0]["end_span"]
    #    ques = self.down.iloc[0]["question"]
    #    answer = self.down.iloc[0]["answer"]
    #    tokenized = self.tokenizer(caption, ques, padding='longest', return_tensors="pt",  add_special_tokens=True)["input_ids"]
      
    #    breakpoint()

    def __getitem__(self, i):
        img_id = self.data.iloc[i]["image_file"]

        caption = self.data.iloc[i]["description"].lower()
        # span_start = self.down.iloc[i]["start_span"]
        # span_end = self.down.iloc[i]["end_span"]
        label = self.down.iloc[i]["label"]
        ques = self.down.iloc[i]["question"]
        answer = self.down.iloc[i]["answer"]
        if self.downstream:
            return label,caption, ques, answer
        path = os.path.join(self.images_path,img_id)

        image = Image.open(path).convert("RGB")
        img_jpg = np.asarray(image)
        # img_jpg = self.hf[str(img_id)][()]
        prompt = "caption en"

        try:
            input_ids, _, image = self.processor(text=prompt, images=img_jpg, return_tensors="pt").values()
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            input_ids, _, image = self.processor(text=prompt, images=rgb_image, return_tensors="pt").values()
        

        return label,caption, ques, answer, image.squeeze(0), img_id, input_ids.squeeze(0)


    def __len__(self):
        return len(self.data)

    
    
    def collate_fn(self, batch):
        
        label = [example[0] for example in batch]
        
        max_len = max(len(size) for size in label)
        label = [example + [0] * (max_len - len(example)) for example in label] #pad the labels to equal size, since they depend on no. tokens
        label = torch.tensor(label)
        caption = [example[1] for example in batch]
        ques = [example[2] for example in batch]
        answer = [example[3] for example in batch]
            # ques = [que + self.tokenizer.sep_token for que in ques]
        # ques = [que + self.tokenizer.sep_token for que in ques]

        # sep_token_id = self.tokenizer.sep_token_id
        # sep_token_position = (tokenized_inputs == sep_token_id).nonzero(as_tuple=True)[1][0].item()
        # attn_mask[sep_token_position::] = 0
        # attn_mask = torch.tensor(attn_mask)
        if self.downstream:
            tokenized = self.tokenizer(ques, caption, padding='longest', return_tensors="pt")
            
            tokenized_input = tokenized["input_ids"]
#pad the labels to be equal to inputs padding
            padding_size = tokenized_input.size(1) - label.size(1)
            label = torch.nn.functional.pad(label, (0, padding_size)) 
            # hot_label = torch.nn.functional.one_hot(label, num_classes=3)
            attn_mask = tokenized["attention_mask"]
            return label, tokenized_input, attn_mask, answer, caption, ques
        images = torch.stack([example[4] for example in batch])
        ids = [[example[5] for example in batch]]
        # prompt = "caption en"
        input_ids = torch.stack([example[6] for example in batch])
        # prompt_list = [prompt for i in range(0,len(ques))]
        # prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]


        return label, answer, caption, ques, images, input_ids, ids




class DCI(Dataset):
    def __init__(self, img_root=None, ann_root=None, split="train", downstream=True):
       self.img_root = "./coco_images.h5"
       self.downstream = downstream
       self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

       if split == "train":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=10000, random_state=42)

       elif split == "finetune":
             self.ann_root = "./my_datasets/annotations/captions_train2017.json"
             self.down_root = "./VQA_datasets/VQA_BIO_train.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=50000)

       else:
             self.ann_root = "./my_datasets/captions_val2014.json"
             self.down_root = "./VQA_datasets/VQA_BIO_val.json"
             self.down = pd.read_json(self.down_root)
             self.down = self.down.sample(n=20000)

    #    self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
       self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-cococap-224")
    #    self.hf = h5py.File(self.img_root, 'r')
    #    self.hf = h5py.File(self.img_root, 'r')
       self.images_path = "./dci/densely_captioned_images/photos"
 
       with open("./dci/densely_captioned_images/splits.json") as f:
            self.data_all = json.load(f)
    #    self.data = pd.read_json("./dci/densely_captioned_images/splits.json")
       self.data = self.data_all["train"] + self.data_all["valid"] + self.data_all["test"]



    #    breakpoint()
    #    img_id = self.data.iloc[0]["image"]
    #    caption = self.data.iloc[0]["caption"]
    #    breakpoint()
    #    span_start = self.down.iloc[0]["start_span"]
    #    span_end = self.down.iloc[0]["end_span"]
    #    ques = self.down.iloc[0]["question"]
    #    answer = self.down.iloc[0]["answer"]
    #    tokenized = self.tokenizer(caption, ques, padding='longest', return_tensors="pt",  add_special_tokens=True)["input_ids"]
      
    #    breakpoint()

    def __getitem__(self, i):
        cap_id = self.data[i]
        path = "./dci/densely_captioned_images/complete/"
        cap_path = os.path.join(path,cap_id)
        with open(cap_path) as f:
            cap_dict = json.load(f)

        caption = cap_dict["summaries"]["base"]
        img_id = cap_dict["image"]

        # span_start = self.down.iloc[i]["start_span"]
        # span_end = self.down.iloc[i]["end_span"]
        label = self.down.iloc[0]["label"]
        ques = self.down.iloc[0]["question"]
        answer = self.down.iloc[0]["answer"]
        if self.downstream:
            return label,caption, ques, answer
        path = os.path.join(self.images_path,img_id)

        image = Image.open(path).convert("RGB")
        img_jpg = np.asarray(image)
        prompt = "caption en"

        try:
            input_ids, _, image = self.processor(text=prompt, images=img_jpg, return_tensors="pt").values()
        except: # A WILD GREY IMAGE APPEARS!!
            rgb_image = np.expand_dims(img_jpg, axis=-1)
            rgb_image = np.repeat(rgb_image, 3, axis=-1)
            input_ids, _, image = self.processor(text=prompt, images=rgb_image, return_tensors="pt").values()
        

        return label,caption, ques, answer, image.squeeze(0), img_id, input_ids.squeeze(0)


    def __len__(self):
        return len(self.data)

    
    
    def collate_fn(self, batch):
        
        label = [example[0] for example in batch]
        
        max_len = max(len(size) for size in label)
        label = [example + [0] * (max_len - len(example)) for example in label] #pad the labels to equal size, since they depend on no. tokens
        label = torch.tensor(label)
        caption = [example[1] for example in batch]
        ques = [example[2] for example in batch]
        answer = [example[3] for example in batch]
            # ques = [que + self.tokenizer.sep_token for que in ques]
        # ques = [que + self.tokenizer.sep_token for que in ques]

        # sep_token_id = self.tokenizer.sep_token_id
        # sep_token_position = (tokenized_inputs == sep_token_id).nonzero(as_tuple=True)[1][0].item()
        # attn_mask[sep_token_position::] = 0
        # attn_mask = torch.tensor(attn_mask)
        if self.downstream:
            tokenized = self.tokenizer(ques, caption, padding='longest', return_tensors="pt")
            
            tokenized_input = tokenized["input_ids"]
#pad the labels to be equal to inputs padding
            padding_size = tokenized_input.size(1) - label.size(1)
            label = torch.nn.functional.pad(label, (0, padding_size)) 
            # hot_label = torch.nn.functional.one_hot(label, num_classes=3)
            attn_mask = tokenized["attention_mask"]
            return label, tokenized_input, attn_mask, answer, caption, ques
        images = torch.stack([example[4] for example in batch])
        ids = [[example[5] for example in batch]]
        # prompt = "caption en"
        input_ids = torch.stack([example[6] for example in batch])
        # prompt_list = [prompt for i in range(0,len(ques))]
        # prompt_list = self.processor.tokenizer(text= [example for example in prompt_list], padding='longest', return_tensors="pt")["input_ids"]


        return label, answer, caption, ques, images, input_ids, ids


