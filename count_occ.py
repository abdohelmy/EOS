import json
with open("/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/Recaption/VQA_BIO_train_roberta.json") as f:
            data = json.load(f)
processed_records = []

count_list = [0,0,0,0,0]

for elm in data:
    ind = elm['label'].count(1)
    count_list[ind]+=1

print(count_list)
breakpoint()



for elm in data:

    start = elm['label'].index(1)

    try:
        end = max(loc for loc, val in enumerate(elm['label']) if val == 2)
        
    except:
        end=start
    # if end != start:
    #     breakpoint()
    new_record = {
    'image_id': elm['image_id'],
    'question_id': elm['question_id'],
    'question' : elm['question'],
    'answer': elm['answer'],
    'start_span': start,
    'end_span' : end,
    'caption': elm['caption']
}
    processed_records.append(new_record)
    
with open('VQA_span_val_squad_q.json', 'w') as f:
    json.dump(processed_records, f, indent=7)


# from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# model_name = "deepset/roberta-base-squad2"
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)