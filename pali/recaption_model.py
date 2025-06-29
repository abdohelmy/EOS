from transformers import AutoProcessor, Blip2ForConditionalGeneration, Blip2Config, Blip2Model
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForVisualQuestionAnswering, AutoModelForCausalLM
from torch import nn
import torch
from typing import Any, Optional, Tuple, Union
from torch.nn import DataParallel as DDP
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model
# from transformers.generation.utils import *

# class bert_recaption(nn.Module):
#     def __init__(self):  
#         super().__init__()


#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
#         self.model = AutoModel.from_pretrained("efederici/sentence-bert-base")
#         # self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#         # self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
#         # self.model.cls.predictions = self.model.cls.predictions.transform
       
#         # self.bert_input = model.bert.embeddings
#         # encoder_layers = self.model.bert.encoder.layer[0:3]
#         # self.model.bert.encoder.layer = encoder_layers
#         # self.model = self.model.bert

#         # #Things dataset
#         # self.downtask1_fc1 = nn.Linear(768, 512)
#         # self.downtask1_relu = nn.ReLU()
#         # self.downtask1_fc2 = nn.Linear(512, 384)
#         # self.downtask1_relu2 = nn.ReLU()
#         # self.downtask1_fc3 = nn.Linear(384, 182)
#         # # self.downtask_fc = nn.Linear(768, 182)
#         # #self.downtask1_sigmoid = nn.Sigmoid()

#         # # #attributes dataset
#         # self.downtask2_fc1 = nn.Linear(768, 512)
#         # self.downtask2_relu = nn.ReLU()
#         # self.downtask2_fc2 = nn.Linear(512, 384)
#         # self.downtask2_relu2 = nn.ReLU()
#         # self.downtask2_fc3 = nn.Linear(384, 204)

#         #Ref_coco dataset
#         # self.downtask3_proj = nn.Linear(768*3, 768)
#         # self.downtask3_relu = nn.ReLU()
#         # self.downtask3_fc1 = nn.Linear(768, 512)
#         # self.downtask3_relu2 = nn.ReLU()
#         # self.downtask3_fc2 = nn.Linear(516, 128)
#         # self.downtask3_relu3 = nn.ReLU()
#         # self.downtask3_fc3 = nn.Linear(128, 4)

#         self.object = nn.Sequential(
#             nn.Linear(768, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 90)
#         )

#         # self.vqa_relu2 = nn.ReLU()
#         # self.vqa_fc3 = nn.Linear(384, 200)
#     def forward(self, input_cap, mask, subj=None, obj=None, sub_c=None):

#         # tokenized = self.tokenizer(input_cap, return_tensors='pt')
#         # x = tokenized["input_ids"]

#         #Using attention mask f
#         embeddings = self.model(input_cap, attention_mask=mask)['pooler_output']
        
#         out = self.object(embeddings)
        
#         #['pooler_output']

#         # #Things dataset
#         # x = self.downtask1_fc1(cap)
#         # x = self.downtask1_relu(x)
#         # x = self.downtask1_fc2(x)
#         # x = self.downtask1_relu2(x)
#         # down1 = self.downtask1_fc3(x)

#         #attributes dataset
#         # x = self.downtask2_fc1(cap)
#         # x = self.downtask2_relu(x)
#         # x = self.downtask2_fc2(x)
#         # x = self.downtask2_relu2(x)
#         # down2 = self.downtask2_fc3(x)



#         # #VQA dataset
#         # x = self.vqa_fc1(cap)
#         # x = self.vqa_relu(x)
#         # out = self.vqa_fc2(x)
#         # x = self.vqa_relu2(x)
#         # x = self.vqa_fc3(x)
#         out = self.object(cap)

#         # out = F.log_softmax(x, dim=1)



#         #Ref_coco dataset
#         # embeddings = self.model(subj)[0]
#         # input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
#         # subj = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         # embeddings = self.model(obj)[0]
#         # input_mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
#         # obj = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         # breakpoint()
#         # x = downtask3_proj(torch.cat((input_cap,subj,obj),1))
#         # x = self.downtask3_relu(x)
#         # x = self.downtask3_fc1(x)
#         # x = self.downtask3_relu2(x)
#         # x = self.downtask3_fc2(torch.cat((x,sub_c),1))
        
#         return out


class bert_recaption(nn.Module):
    def __init__(self):  
        super().__init__()

        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
   
        # self.bert_input = model.bert.embeddings
        # encoder_layers = self.model.bert.encoder.layer[0:3]
        # self.model.bert.encoder.layer = encoder_layers
        # self.model = self.model.bert

        self.vqa = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 100)
        )


        #Things dataset
        # self.downtask1_fc1 = nn.Linear(768, 512)
        # self.downtask1_relu = nn.ReLU()
        # self.downtask1_fc2 = nn.Linear(512, 384)
        # self.downtask1_relu2 = nn.ReLU()
        # self.downtask1_fc3 = nn.Linear(384, 182)
        # self.downtask_fc = nn.Linear(768, 182)
        #self.downtask1_sigmoid = nn.Sigmoid()

        # #attributes dataset
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
    def forward(self, input_cap, ques, subj=None, obj=None, sub_c=None):

        # tokenized = self.tokenizer(input_cap, return_tensors='pt')
        # x = tokenized["input_ids"]

        #Using attention mask f
        inputs = [c + " " + q for q,c in zip(ques,input_cap)]
  
        embeddings = self.model.encode(inputs)
       # breakpoint()
        device = self.model.device
        out = self.vqa(torch.tensor(embeddings).to(self.model.device))

        #Things dataset
        # x = self.downtask1_fc1(embeddings)
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
        
        return out



class blip_recaption(Blip2ForConditionalGeneration):
    def __init__(self): 
        
        configuration = Blip2Config(vision_config = {"image_size": 364})
       
        super().__init__(configuration)
       
        #self.model_blip = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir = "./new_blip2")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        self.model_blip = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco", cache_dir = "/l/users/israfel.salazar/abdo/blip2_coco",)
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        # opt = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b")
        # breakpoint()
        # if lora:
        #     config = LoraConfig(
        #         r=16,
        #         lora_alpha=32,
        #         lora_dropout=0.05,
        #         bias="none",
        #         target_modules=["q_proj", "k_proj"]
        #     )

        #     self.model_blip = get_peft_model(self.model_blip, config)
            
 #### The pretrained coco one have different input and causing me headache, copying the weights and changing the positional embeddings layer back
 ### A terrible hack but works with better cider than wtihout CIDEr Score: 1.0602832101535398
        # breakpoint()
        # z = self.model_blip.vision_model.embeddings.position_embedding
        # self.model_blip.vision_model.embeddings.position_embedding = self.model2.vision_model.embeddings.position_embedding
        # self.model_blip.load_state_dict(self.model2.state_dict(), strict=True)
        # self.model_blip.vision_model.embeddings.position_embedding = z
        # del self.model2
       # self.model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", torch_dtype=torch.float16)
    
    # def generate(self, image, caption, prompts, ids):
    #     processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    #     generated_ids = self.model.generate(input_ids=prompts, pixel_values=image, max_length=60)
    #     generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
    #     captions = {key: [value] for key,value in zip(ids[0], generated_caption)}

    #     return captions


    # def forward(self, image, caption, prompts, ids=None):
        
    #     output = self.model(input_ids=prompts ,pixel_values=image, labels=caption)
    # #     breakpoint()
    # #     output = self.model(input_ids=prompts ,pixel_values=image.half())

    # #     # qformer = output["qformer_outputs"]["pooler_output"]
    # #     # qformer_outputs = self.model.get_qformer_features(image.half())['pooler_output']
    # #     # # prompts_embeded = self.model.language_model.model.decoder.embed_tokens(prompts)
    # #     # qformer_embeded = self.model.language_projection(qformer_outputs).unsqueeze(1)
    # #     # inputs = torch.cat((prompts_embeded,qformer_embeded), 1)
    # #     # lang = nn.Sequential(*list(self.model.language_model.model.decoder.children())[2:]) 
    # #     # breakpoint()
    #     return output
    @torch.enable_grad()
    def Hope(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.model.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)


        output_sequences = input_ids
        self.temperature = 1
        probs = None
        tokens = None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if self.config.use_decoder_only_language_model:
            while True:    
                outputs = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                logits = outputs.logits 
                
               
                logits = logits[:, -1, :] / (

                        self.temperature if self.temperature > 0 else 1.0

                    )

                prob = torch.nn.functional.softmax(logits, dim=-1)
                
                next_token = torch.multinomial(prob, num_samples=1)
     
                words_prob = torch.gather(prob, dim=1, index=next_token).squeeze(-1)
                output_sequences = torch.cat([output_sequences, next_token], dim=-1)
                for tok in next_token:
                    
                    if tok.item() != self.processor.tokenizer.eos_token_id :

                        tokens = tok
                
                if tokens is None or len(output_sequences[0])>12: #not sure how to handle the eos in a batch, a temporary hack till things workout
                    break
                tokens = None
                next_Token_embeds = self.model.language_model.get_input_embeddings()(next_token)
                inputs_embeds = torch.cat((inputs_embeds, next_Token_embeds), dim=1)
                #inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
                attention_mask = torch.ones_like(output_sequences)
                expected_device = language_model_attention_mask.device
                attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
                # breakpoint()
                if probs == None:
                    probs = words_prob.unsqueeze(1)
                else:
                    probs = torch.cat((probs,words_prob.unsqueeze(1)), dim=1)
             


        return  probs, output_sequences[:,2:]


    @torch.enable_grad()
    def my_generate(self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor
    ):
        
        return_dict = True
        attention_mask = None
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.model.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )


        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
        outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # output_attentions=output_attentions,
                # output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # breakpoint()
        logits = outputs.logits if return_dict else outputs[0]
    
    # def my_generate(
     
    #     self,
    #     pixel_values: torch.FloatTensor,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.LongTensor] = None,
    #     **generate_kwargs,):

    #     """
    #     Overrides `generate` function to be able to use the model as a conditional generator.

    #     Args:
    #         pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
    #             Input images to be processed.
    #         input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
    #             The sequence used as a prompt for the generation.
    #         attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
    #             Mask to avoid performing attention on padding token indices

    #     Returns:
    #         captions (list): A list of strings of length batch_size * num_captions.
    #     """
    #     if hasattr(self, "hf_device_map"):
    #         # preprocess for `accelerate`
    #         self._preprocess_accelerate()

    #     batch_size = pixel_values.shape[0]
    #     image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
    #     image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    #     query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    #     query_outputs = self.qformer(
    #         query_embeds=query_tokens,
    #         encoder_hidden_states=image_embeds,
    #         encoder_attention_mask=image_attention_mask,
    #         return_dict=True,
    #     )
    #     query_output = query_outputs.last_hidden_state

    #     language_model_inputs = self.language_projection(query_output)
    #     language_attention_mask = torch.ones(
    #         language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
    #     )
    #     if input_ids is None:
    #         input_ids = (
    #             torch.LongTensor([[self.config.text_config.bos_token_id]])
    #             .repeat(batch_size, 1)
    #             .to(image_embeds.device)
    #         )
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids)
    #     attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

    #     # concatenate query embeddings with prompt embeddings
    #     inputs_embeds = self.get_input_embeddings()(input_ids)
    #     inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

    #     # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
    #     if not self.language_model.config.is_encoder_decoder:
    #         generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1]
    #         generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]
    #     with torch.enable_grad():
    #         outputs = self.language_model.generate(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             **generate_kwargs,
    #         )

    #     return outputs