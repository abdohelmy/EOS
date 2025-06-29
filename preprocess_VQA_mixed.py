import json
from collections import Counter
import random
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaTokenizer

def preprocess_json(input_file, output_records_file, output_mapping_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        records = json.load(f)

    # Collect all answers (converted to lowercase) across the dataset
    # all_answers = []
    # queses = []
    # ques_type = ["what color is the", "what color are the", "what color is", "what color", "what is", "what is the", "what is", "what is on the", "what type of", "what are they"]
    # #"what is", "what is the", "what is", "what is on the", "what type of", "what are they" 
    # for record in records["annotations"]:
    #     for ans in record['answers']:
    #         if ans['answer_confidence'] != 'no':  # Only consider answers with confidence other than 'no'
    #             answer_lower = ans['answer'].lower()
    #             # Save only answers with two or less words
    #             if len(answer_lower.split()) <= 2:
    #                 if record["question_type"] in ques_type:
                        
    #                     all_answers.append(answer_lower)

    # tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    with open("./my_datasets/v2_mscoco_train2014_annotations.json", 'r') as f:
        records_train = json.load(f)
#, "what is the", "what is on the", "what type of", "what are they" 
    # Collect all answers (converted to lowercase) across the dataset
    train_vqa = []
    ques_type = ["what color is the", "what color are the", "what color is", "what color", "what is", "what is the", "what is", "what is on the", "what type of", "what are they"]
    
    for record in records_train["annotations"]:
        for ans in record['answers']:
            if ans['answer_confidence'] != 'no':  # Only consider answers with confidence other than 'no'
                answer_lower = ans['answer'].lower()
                if answer_lower!= "yes" and answer_lower != "no" and answer_lower.isalpha() and len(answer_lower)<6:
#                           
                            train_vqa.append(record)
                            break

    processed_records = []
    with open("./my_datasets/annotations/captions_train2017.json") as f:
            data = json.load(f)
            data = data["annotations"]
            data = pd.DataFrame(data)
    with open("./my_datasets/v2_OpenEnded_mscoco_train2014_questions.json") as f:
         questions = json.load(f)["questions"]
    image_question_dict = {} #create dictonary to have questions with image as key
    for record in questions:
   
        image_id = str(record['image_id'])
        question = record['question']
        question_id = record['question_id']
        image_question_dict[question_id] = question
    for record in train_vqa:
        
        # Initialize a vector of zeros for each frequent answer
        # answer_vector = [0.0] * len(picked_answers)
        # Set to 1 for answers present in this record, converting them to lowercase
        captions = data[data["image_id"]==record['image_id']]["caption"]
        for cap in captions:
            ques = image_question_dict[record['question_id']]
            token_cap = tokenizer(ques.lower(), cap.lower())['input_ids']
            # token_cap = tokenizer(cap.lower())['input_ids']
            label = [0] * len(token_cap)
            correct_answers = []
            for ans in record['answers']:
                flag = 0
                if ans['answer_confidence'] == 'yes':
                    answer_lower = ans['answer'].lower()
                    token_answer =  tokenizer(" "+answer_lower)['input_ids'][1:-1]
                    # Check if answer is in the frequent list and has two or less words
                    # if ans['answer_confidence'] == 'yes' and answer_lower in picked_answers and len(answer_lower.split()) <= 2:
                #         for t in ques_type:
                #                 if record["question_type"].lower() == t.lower():
                #                     answer_index = answer_to_index[answer_lower]
                #                     answer_vector[answer_index] = 1.0
                #                     flag = 1
                #                     break
                # # Create the new record
                # if flag == 1:

                    # breakpoint()
                    #caption = captions.iloc[random.randint(len(captions)-1)]["caption"]
                    # if answer_lower in cap.lower():
                    #     breakpoint()
                    if set(token_answer).issubset(token_cap) :
                        # breakpoint()
                        
                        correct_answers.append(answer_lower)
                        flag = 0
                        for tok in token_answer:
                           
                                
                                index = max(loc for loc, val in enumerate(token_cap) if val == tok)
                                sep_token_id = tokenizer.sep_token_id
                                sep_token_position = token_cap.index(sep_token_id)
                                # breakpoint()
                                if index < sep_token_position:
                                    continue
                                
                                if flag ==0:
                                    label[index] = 1
                                    flag = 1
                                else:
                                    label[index] = 2

                        #     # breakpoint()
                        #     cap_split = cap.lower().split(" ")
                        #     ind = random.randint(3,len(cap_split)-1)
                        #     cap_augmented = cap_split[0:ind] + [answer_lower] + cap_split[ind:]
                        #     cap = " ".join(cap_augmented)
            if 1 in label:
                # breakpoint()        
                new_record = {
                    'image_id': record['image_id'],
                    'question_id': record['question_id'],
                    'question' : image_question_dict[record['question_id']],
                    'answer': correct_answers,
                    'label': label,
                    'caption': cap
                }
                processed_records.append(new_record)
                label = [0]
                # breakpoint()
            
    with open('VQA_BIO_train_roberta.json', 'w') as f:
        json.dump(processed_records, f, indent=6)


            # flag = 0

#     # Count occurrences of each answer and filter out those occurring three times or fewer
# #     import operator
# #     random.seed(42)
# #     answer_counts = Counter(all_answers_train)
# #   #  frequent_answers = {answer for answer, count in answer_counts.items() if count > 200}
# #     # breakpoint()
# #     # picked_answers = random.sample(frequent_answers,200)
# #     picked_answers = dict(sorted(answer_counts.items(), key=operator.itemgetter(1), reverse=True)[:100]) #kolo ya waleed, enta 3amel comment foo2
# #     # Create a mapping from frequent answer to index
# #     answer_to_index = {answer: index for index, answer in enumerate(sorted(picked_answers))}
# #     processed_records = []
#     with open("./my_datasets/v2_OpenEnded_mscoco_val2014_questions.json") as f:
#          questions = json.load(f)["questions"]
#     image_question_dict = {} #create dictonary to have questions with image as key
#     with open("./my_datasets/captions_val2014.json") as f:
#             data = json.load(f)
#             data = data["annotations"]
#             data = pd.DataFrame(data)
#     for record in questions:
#     # Ensure the 'image_id' is treated as a string to avoid JSON integer key issues
        
#         image_id = str(record['image_id'])
#         question = record['question']
#         question_id = record['question_id']
#         image_question_dict[question_id] = question
#     for record in records["annotations"]:
#         flag = 0
#         # Initialize a vector of zeros for each frequent answer
#         answer_vector = [0.0] * len(picked_answers)
#         # Set to 1 for answers present in this record, converting them to lowercase
#         for ans in record['answers']:
#             answer_lower = ans['answer'].lower()
#             # Check if answer is in the frequent list and has two or less words
#             if ans['answer_confidence'] == 'yes' and answer_lower in picked_answers and len(answer_lower.split()) <= 2:
#                     for t in ques_type:
#                         if record["question_type"].lower() == t.lower():
#                             answer_index = answer_to_index[answer_lower]
#                             answer_vector[answer_index] = 1.0
#                             flag = 1
#                             break
                
#         # # Create the new record
#         if flag == 1:
#             captions = data[data["image_id"]==record['image_id']]["caption"]
#             for cap in captions:
#                 # breakpoint()
#                 #caption = captions.iloc[random.randint(len(captions)-1)]["caption"]
#                 if answer_lower not in cap.lower():
#                     continue
#                 #     cap_split = cap.lower().split(" ")
#                 #     ind = random.randint(3,len(cap_split)-1)
#                 #     cap_augmented = cap_split[0:ind] + [answer_lower] + cap_split[ind:]
#                 #     cap = " ".join(cap_augmented)
                    
#                 new_record = {
#                     'image_id': record['image_id'],
#                     'question_id': record['question_id'],
#                     'question' : image_question_dict[record['question_id']],
#                     'answer_vector': answer_vector,
#                     'caption': cap
#                 }
#                 processed_records.append(new_record)

    with open("./my_datasets/v2_mscoco_val2014_annotations.json", 'r') as f:
        records_val = json.load(f)
#, "what is the", "what is on the", "what type of", "what are they" 
    # Collect all answers (converted to lowercase) across the dataset
    val_vqa = []
    ques_type = ["what color is the", "what color are the", "what color is", "what color", "what is", "what is the", "what is", "what is on the", "what type of", "what are they"]
    
    for record in records_val["annotations"]:
        for ans in record['answers']:
            if ans['answer_confidence'] != 'no':  # Only consider answers with confidence other than 'no'
                answer_lower = ans['answer'].lower()
                if answer_lower!= "yes" and answer_lower != "no" and answer_lower.isalpha() and len(answer_lower)<6:
#                           
                            val_vqa.append(record)
                            break

    processed_records = []
    with open("./my_datasets/v2_OpenEnded_mscoco_val2014_questions.json") as f:
         questions = json.load(f)["questions"]
    image_question_dict = {} #create dictonary to have questions with image as key
    with open("./my_datasets/captions_val2014.json") as f:
            data = json.load(f)
            data = data["annotations"]
            data = pd.DataFrame(data)
    for record in questions:
    # Ensure the 'image_id' is treated as a string to avoid JSON integer key issues
        
        image_id = str(record['image_id'])
        question = record['question']
        question_id = record['question_id']
        image_question_dict[question_id] = question
    for record in val_vqa:
        
        # Initialize a vector of zeros for each frequent answer
        # answer_vector = [0.0] * len(picked_answers)
        # Set to 1 for answers present in this record, converting them to lowercase
        captions = data[data["image_id"]==record['image_id']]["caption"]
        for cap in captions:
            correct_answers = []
            ques = image_question_dict[record['question_id']]
            token_cap = tokenizer(ques.lower(), cap.lower())['input_ids']
            # token_cap = tokenizer(cap.lower())['input_ids']
            label = [0] * len(token_cap)
            for ans in record['answers']:
                flag = 0
                if ans['answer_confidence'] == 'yes':
                    answer_lower = ans['answer'].lower()
                    token_answer =  tokenizer(" "+answer_lower)['input_ids'][1:-1]
                    # Check if answer is in the frequent list and has two or less words
                    # if ans['answer_confidence'] == 'yes' and answer_lower in picked_answers and len(answer_lower.split()) <= 2:
                #         for t in ques_type:
                #                 if record["question_type"].lower() == t.lower():
                #                     answer_index = answer_to_index[answer_lower]
                #                     answer_vector[answer_index] = 1.0
                #                     flag = 1
                #                     break
                # # Create the new record
                # if flag == 1:


                    # breakpoint()
                    #caption = captions.iloc[random.randint(len(captions)-1)]["caption"]
                    # if answer_lower in cap.lower():
                    #     breakpoint()
                    if set(token_answer).issubset(token_cap):
                        correct_answers.append(answer_lower)
                        
                        
                        
                        flag = 0
                        for tok in token_answer:

                                index = max(loc for loc, val in enumerate(token_cap) if val == tok)
                                sep_token_id = tokenizer.sep_token_id
                                sep_token_position = token_cap.index(sep_token_id)
                                # breakpoint()
                                if index < sep_token_position:
                                    continue
                                
                                if flag ==0:
                                    label[index] = 1
                                    flag = 1
                                else:
                                    label[index] = 2

                        #     # breakpoint()
                        #     cap_split = cap.lower().split(" ")
                        #     ind = random.randint(3,len(cap_split)-1)
                        #     cap_augmented = cap_split[0:ind] + [answer_lower] + cap_split[ind:]
                        #     cap = " ".join(cap_augmented)
            if 1 in label:
                # breakpoint()        
                new_record = {
                    'image_id': record['image_id'],
                    'question_id': record['question_id'],
                    'question' : image_question_dict[record['question_id']],
                    'answer': correct_answers,
                    'label': label,
                    'caption': cap
                }
                processed_records.append(new_record)
                label = [0]
    # Write the processed records to the output JSON file
    with open(output_records_file, 'w') as f:
        json.dump(processed_records, f, indent=6)

    # Additionally, save the answer_to_index mapping

    
    # Write the processed records to the output JSON file


# Replace 'input.json', 'output_records.json', and 'output_mapping.json' with your actual file paths
preprocess_json('/fsx/homes/Abdelrahman.Mohamed@mbzuai.ac.ae/abdo/Recaption/my_datasets/v2_mscoco_val2014_annotations.json', 'VQA_BIO_val_roberta.json', 'really_what_only_100_binary_mapping_val_vector_gold.json')
