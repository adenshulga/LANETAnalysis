import json
import os

def convert_dataset(original_dataset):
    converted_data = {'train': [], 'validate': [], 'test': []}  
    
    for section in original_dataset:
        for user_id, baskets in original_dataset[section].items():
            user_baskets = []
            set_time = 1  
            for basket in baskets:
                user_baskets.append({
                    "user_id": user_id,  
                    "items_id": basket,  
                    "set_time": set_time 
                })
                set_time += 1  

            converted_data[section].append(user_baskets)

    return converted_data

# Example

# dataset_name = 'TMS'
# dnntsp_format_path = f"/app/DNNTSP/data/{dataset_name}/{dataset_name}.json"

# os.makedirs(f'dataset/{dataset_name}', exist_ok=True )

# with open(dnntsp_format_path, 'r') as f:
#     dataset = json.load(f)

# converted_dataset = convert_dataset(dataset)

# with open(f'dataset/{dataset_name}/{dataset_name}.json', 'w') as f :
#     json.dump(converted_dataset, f)





