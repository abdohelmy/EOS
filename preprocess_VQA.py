import json
from collections import Counter
import random
def preprocess_json(input_file, output_records_file, output_mapping_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        records = json.load(f)

    # Collect all answers (converted to lowercase) across the dataset
    all_answers = []
    ques_type = ["what color is the", "what color are the", "what color is", "what color", "what is", "what is the", "what is", "what is on the", "what type of", "what are they" ]
    
    for record in records["annotations"]:
        for ans in record['answers']:
            if ans['answer_confidence'] != 'no':  # Only consider answers with confidence other than 'no'
                answer_lower = ans['answer'].lower()
                # Save only answers with two or less words
                if len(answer_lower.split()) <= 2:
                    if record["question_type"] in ques_type:
                        all_answers.append(answer_lower)

    # Count occurrences of each answer and filter out those occurring three times or fewer

    random.seed(42)
    answer_counts = Counter(all_answers)
    frequent_answers = {answer for answer, count in answer_counts.items() if count > 5}
    picked_answers = random.sample(frequent_answers,1000)
    # Create a mapping from frequent answer to index
   
    answer_to_index = {answer: index for index, answer in enumerate(sorted(picked_answers))}

    processed_records = []
    with open("/home/israfel.salazar/abdo/Recaption/my_datasets/v2_OpenEnded_mscoco_train2014_questions.json") as f:
         questions = json.load(f)["questions"]
    image_question_dict = {} #create dictonary to have questions with image as key
    for record in questions:
    # Ensure the 'image_id' is treated as a string to avoid JSON integer key issues
        image_id = str(record['image_id'])
        question = record['question']
        image_question_dict[image_id] = question
    for record in records["annotations"]:
        # flag = 0
        # Initialize a vector of zeros for each frequent answer
        answer_vector = [0] * len(picked_answers)
        # Set to 1 for answers present in this record, converting them to lowercase
        for ans in record['answers']:
            answer_lower = ans['answer'].lower()
            # Check if answer is in the frequent list and has two or less words
            if ans['answer_confidence'] != 'no' and answer_lower in picked_answers and len(answer_lower.split()) <= 2:
                answer_index = answer_to_index[answer_lower]
                answer_vector[answer_index] = 1
                # flag = 1
                # break
        # Create the new record
        # if flag == 1:
        new_record = {
            'image_id': record['image_id'],
            'question_id': record['question_id'],
            'question' : image_question_dict[str(record['image_id'])],
            'answer_vector': answer_vector
        }
        processed_records.append(new_record)
            # flag = 0

    # Write the processed records to the output JSON file
    with open(output_records_file, 'w') as f:
        json.dump(processed_records, f, indent=5)

    # Additionally, save the answer_to_index mapping
    with open(output_mapping_file, 'w') as f:
        json.dump(answer_to_index, f, indent=4)

# Replace 'input.json', 'output_records.json', and 'output_mapping.json' with your actual file paths
preprocess_json('/home/israfel.salazar/abdo/Recaption/my_datasets/v2_mscoco_train2014_annotations.json', 'what_1000_train_multi.json', 'what_1000_mapping_train_multi.json')
