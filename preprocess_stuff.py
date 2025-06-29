import json
from collections import defaultdict
import pandas as pd
import random
# Function to transform the data
def transform_data(input_file1, output_file, total_categories,input_file2=None):
    # Create a default dictionary to hold the mapping of image_id to category vectors
    # transformed_data = defaultdict(lambda: [0.0] * total_categories)
    with open("/home/israfel.salazar/abdo/Recaption/my_datasets/annotations/captions_train2017.json") as f:
            t_data = json.load(f)
            t_data = t_data["annotations"]
    #         # t_data = pd.DataFrame(t_data)
   # with open("/home/israfel.salazar/abdo/Recaption/my_datasets/captions_val2014.json") as f:
         #   t_data = json.load(f)
           # t_data = t_data["annotations"]
            # v_data = pd.DataFrame(v_data)
#region
    items = {
    0: 'unlabeled',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'street sign',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    26: 'hat',
    27: 'backpack',
    28: 'umbrella',
    29: 'shoe',
    30: 'eye glasses',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    45: 'plate',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    66: 'mirror',
    67: 'dining table',
    68: 'window',
    69: 'desk',
    70: 'toilet',
    71: 'door',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    83: 'blender',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}
#endregion
    dict ={}
    # Read the input JSON file

    with open(input_file1, 'r') as f:
        records = json.load(f)
    pandords = pd.DataFrame(records["annotations"])
    # Iterate through the records and update the category vectors for each image_id
    processed_records = []
    for record in t_data:
        object_vector = [0.0] * total_categories
        # breakpoint()
        image_id = record['image_id']
        cap = record["caption"]
        try:
            category_ids = pandords[pandords["image_id"]==record['image_id']]["category_id"]
        except:
            continue
        # category_id = record['category_id']
        

        # Subtract 1 from category_id to convert to 0-indexed for the vector position
        
        for cat in category_ids:
                if cat ==1:
                         
                         continue
                object_vector[cat] = 1.0
                
                # if items[cat] not in cap.lower():
                #     cap_split = cap.lower().split(" ")
                #     ind = random.randint(3,len(cap_split)-1)
                #     cap_augmented = cap_split[0:ind] + [items[cat]] + cap_split[ind:]
                #     cap = " ".join(cap_augmented)
                    
        # try:
        #     dict[str(category_id)]+=1
        # except:
        #     dict[str(category_id)] = 1
        # transformed_data[image_id][category_id - 1] = 1
        new_record = {'image_id': record['image_id'],
                        'id' : record['id'],
                        'object': object_vector,
                        'caption' : cap
                    }
        processed_records.append(new_record)
  
    with open(output_file, 'w') as f:
        json.dump(processed_records, f, indent=4)




















    # with open(input_file1, 'r') as f:
    #     records = json.load(f)
    # pandords = pd.DataFrame(records["annotations"])
    # # Iterate through the records and update the category vectors for each image_id
    # processed_records = []
    # for record in records["annotations"]:
    #     object_vector = [0.0] * total_categories
    #     # breakpoint()
    #     image_id = record['image_id']
    #     category_id = record['category_id']
    #     breakpoint()
    #     if category_id ==1:
    #         continue
    #     # Subtract 1 from category_id to convert to 0-indexed for the vector position
    #     object_vector[category_id - 1] = 1
    #     try:
    #         captions = t_data[t_data["image_id"]==record['image_id']]["caption"]
    #     except:
    #         captions = v_data[v_data["image_id"]==record['image_id']]["caption"]
    #         for cap in captions:
    #             breakpoint()
    #             caption = captions.iloc[random.randint(len(captions)-1)]["caption"]
    #             if items[category_id - 1] not in cap.lower():
    #                 cap_split = cap.lower().split(" ")
    #                 ind = random.randint(3,len(cap_split)-1)
    #                 cap_augmented = cap_split[0:ind] + [answer_lower] + cap_split[ind:]
    #                 cap = " ".join(cap_augmented)
                    
    #     # try:
    #     #     dict[str(category_id)]+=1
    #     # except:
    #     #     dict[str(category_id)] = 1
    #     # transformed_data[image_id][category_id - 1] = 1
    #     new_record = {'image_id': record['image_id'],
    #                     'id' : record['id'],
    #                     'object': object_vector
    #                 }
    #     processed_records.append(new_record)
    # with open(output_file, 'w') as f:
    #     json.dump(processed_records, f, indent=3)


    # with open(input_file2, 'r') as f:
    #     records = json.load(f)
        
    # # Iterate through the records and update the category vectors for each image_id
    # for record in records["annotations"]:
    #     image_id = record['image_id']
    #     category_id = record['category_id']
    #     if category_id ==183 or category_id ==0:
    #         continue
    #     # Subtract 1 from category_id to convert to 0-indexed for the vector position
    #     transformed_data[image_id][category_id - 1] = 1       
        
    # Write the transformed data to the output JSON file
    # with open(output_file, 'w') as f:
    #     json.dump([{ 'image_id': key, 'labels': value } for key, value in transformed_data.items()], f, indent=4)

# Replace 'input.json' with the path to your input file and 'output.json' with your desired output file name
# Update 182 to the actual number of category_ids you have
transform_data("/home/israfel.salazar/abdo/Recaption/annotations/instances_train2017.json", 'objects_train_multi_raw.json', 91)
