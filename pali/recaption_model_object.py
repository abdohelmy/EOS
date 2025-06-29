from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForVisualQuestionAnswering
from torch import nn
import torch
import torch.nn.functional as F
# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class bert_recaption(nn.Module):
    def __init__(self):  
        super().__init__()


        #self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        #self.model = AutoModelForMaskedLM.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        # self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') #You can loop through parameters name if you want to use LoRA


        self.tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        breakpoint()
        # self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        # self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
        # self.model.cls.predictions = self.model.cls.predictions.transform
       
        # self.bert_input = model.bert.embeddings
        # encoder_layers = self.model.bert.encoder.layer[0:3]
        # self.model.bert.encoder.layer = encoder_layers
        # self.model = self.model.bert

        # #Things dataset
        # self.downtask1_fc1 = nn.Linear(768, 512)
        # self.downtask1_relu = nn.ReLU()
        # self.downtask1_fc2 = nn.Linear(512, 384)
        # self.downtask1_relu2 = nn.ReLU()
        # self.downtask1_fc3 = nn.Linear(384, 182)
        # # self.downtask_fc = nn.Linear(768, 182)
        # #self.downtask1_sigmoid = nn.Sigmoid()

        # # #attributes dataset
        # self.downtask2_fc1 = nn.Linear(768, 512)
        # self.downtask2_relu = nn.ReLU()
        # self.downtask2_fc2 = nn.Linear(512, 384)
        # self.downtask2_relu2 = nn.ReLU()
        # self.downtask2_fc3 = nn.Linear(384, 204)

        #Ref_coco dataset
        # self.downtask3_proj = nn.Linear(768*3, 768)
        # self.downtask3_relu = nn.ReLU()
        # self.downtask3_fc1 = nn.Linear(768, 512)
        # self.downtask3_relu2 = nn.ReLU()
        # self.downtask3_fc2 = nn.Linear(516, 128)
        # self.downtask3_relu3 = nn.ReLU()
        # self.downtask3_fc3 = nn.Linear(128, 4)

        self.span = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # self.object = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 91)
        # )


        # self.vqa_relu2 = nn.ReLU()
        # self.vqa_fc3 = nn.Linear(384, 200)
    def forward(self, input_cap, device, subj=None, obj=None, sub_c=None):

        # tokenized = self.tokenizer(input_cap, return_tensors='pt')
        # x = tokenized["input_ids"]

        #Using attention mask f
        # token_embeddings = self.model(input_cap, attention_mask=mask)[0] #First element of model_output contains all token embeddings
        # input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        #breakpoint()
        #embeddings = self.model(input_cap, attention_mask=mask)['pooler_output']

        
        embeddings = self.model.encode(input_cap)
     
        out = self.object(torch.tensor(embeddings).to(device))
        
        #['pooler_output']

        # #Things dataset
        # x = self.downtask1_fc1(cap)
        # x = self.downtask1_relu(x)
        # x = self.downtask1_fc2(x)
        # x = self.downtask1_relu2(x)
        # down1 = self.downtask1_fc3(x)

        #attributes dataset
        # x = self.downtask2_fc1(cap)
        # x = self.downtask2_relu(x)
        # x = self.downtask2_fc2(x)
        # x = self.downtask2_relu2(x)
        # down2 = self.downtask2_fc3(x)



        # #VQA dataset
        # x = self.vqa_fc1(cap)
        # x = self.vqa_relu(x)
        # out = self.vqa_fc2(x)
        # x = self.vqa_relu2(x)
        # x = self.vqa_fc3(x)
        # out = self.object(cap)

        # out = F.log_softmax(x, dim=1)



        #Ref_coco dataset
        # embeddings = self.model(subj)[0]
        # input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
        # subj = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # embeddings = self.model(obj)[0]
        # input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
        # obj = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # breakpoint()
        # x = downtask3_proj(torch.cat((input_cap,subj,obj),1))
        # x = self.downtask3_relu(x)
        # x = self.downtask3_fc1(x)
        # x = self.downtask3_relu2(x)
        # x = self.downtask3_fc2(torch.cat((x,sub_c),1))
        
        return out.squeeze(-1)



class blip_recaption(nn.Module):
    def __init__(self): 
        super().__init__()
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, cache_dir = "./blip2")
        #self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco", torch_dtype=torch.float16, cache_dir = "./blip2")
        #self.model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", torch_dtype=torch.float16)
    
    # def generate(self, image, caption, prompts, ids):
    #     processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    #     generated_ids = self.model.generate(input_ids=prompts, pixel_values=image, max_length=60)
    #     generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    #     captions = {key: [value] for key,value in zip(ids[0], generated_caption)}

    #     return captions
    def forward(self, image, caption, prompts, ids):
        output = self.model(input_ids=prompts ,pixel_values=image.half(), labels=caption)
        return output
