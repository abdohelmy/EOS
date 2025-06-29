import json
from collections import Counter
import random
def preprocess_json(input_file, output_records_file, output_mapping_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        records = json.load(f)

    processed_records = []
    captions = []
    man = 0
    woman = 0
    dog = 0
    cat = 0
    questions = ["is there a dog?", "is there a woman?", "is there a cat?"]
    for record in records["annotations"]:
        # if "man" in record["caption"].lower().split(" "):
        #     man+=1
        flag = -1
        if "woman" in record["caption"].lower().split(" "):
            woman+=1
            flag = 1
        if "dog" in record["caption"].lower().split(" "):
            dog+=1
            flag = 0
        if "cat" in record["caption"].lower().split(" "):
            cat+=1
            flag = 2
        # flag = 0
        # Initialize a vector of zeros for each frequent answer
        
        # Set to 1 for answers present in this record, converting them to lowercase

            # Check if answer is in the frequent list and has two or less words

                # flag = 1
                # break
        # Create the new record
        # if flag == 1:
        if flag !=-1:
            new_record = {
                'image_id': record['image_id'],
                'caption': record['caption'],
                'question' : questions[flag],
                'answer_vector': 1
            }
            processed_records.append(new_record)
            flag_list = [0,1,2]
            z = random.choice([ele for ele in flag_list if ele != flag])
            new_record = {
                'image_id': record['image_id'],
                'caption': record['caption'],
                'question' : questions[z],
                'answer_vector': 0
            }
            processed_records.append(new_record)
            # flag = 0

    # Write the processed records to the output JSON file
    with open(output_records_file, 'w') as f:
        json.dump(processed_records, f, indent=5)



# Replace 'input.json', 'output_records.json', and 'output_mapping.json' with your actual file paths
preprocess_json('/home/israfel.salazar/abdo/Recaption/annotations/captions_train2017.json', 'dummy_train.json', None)
