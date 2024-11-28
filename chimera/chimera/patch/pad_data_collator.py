import numpy as np
import torch
import pdb

IGNORE_INDEX = -100


def pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids') and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


def concat_pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]

        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids

        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        
        # feat['attention_mask'] = feat['input_ids'].ne(pad_id)
        
        temp_attention_mask = torch.BoolTensor([False] * max_item_length)
        temp_attention_mask[:feat['attention_mask'].shape[0]] = feat['attention_mask']
        feat['attention_mask'] = temp_attention_mask
        


    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags', 'thumbnail', 'sci_domain_ids', "sci_encoder_pixel_value_list" , "sci_encoder_attention_mask_list") and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

        if k in ('pixel_values', 'image_flags', 'thumbnail', 'sci_domain_ids'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])

        
        if k in ("sci_encoder_pixel_value_list" , "sci_encoder_attention_mask_list"):
            num_sci_encoder = len(first["sci_encoder_pixel_value_list"])
            batch[k] = []
            for i in range(num_sci_encoder):
                elements = [f[k][i] for f in features]
                if elements[0] is None:
                    assert  all(element is None for element in elements)
                    batch[k].append(None)
                else:   
                    batch[k].append(torch.concat(elements))
            
            # for i in range(num_sci_encoder):
            #     elements = [f[k][i] for f in features if f[k][i] is not None]
            #     if len(elements)==0:
            #         batch[k].append(None)
            #     else:   
            #         batch[k].append(torch.concat(elements))
        

    # def print_progress_bar(tensor_list):
    #     bar_name = {
    #         "-100":"Text",
    #         "0":"General",
    #         "1":"Table",
    #         "2":"Math",
    #         "3":"Chart"
    #     }
    #     total_sum = sum([t.item() for t in tensor_list])
    #     if total_sum == 0:
    #         return
        
    #     for i, value in enumerate(tensor_list):
    #         proportion = value.item() / total_sum
    #         bar_length = int(proportion * 50)  # 进度条长度为50个字符
    #         bar = '=' * bar_length + ' ' * (50 - bar_length)
    #         # print(f"{bar_name[str(value.item())]}: [{bar}] {proportion:.2%}")
    #         print(f"Element {i + 1}: [{bar}] {proportion:.2%}, {value.item()}/{total_sum}")

            
    # print_progress_bar([torch.sum(batch['sci_domain_ids']==-100),\
    #                         torch.sum(batch['sci_domain_ids']==0),\
    #                         torch.sum(batch['sci_domain_ids']==1),\
    #                         torch.sum(batch['sci_domain_ids']==2),\
    #                         torch.sum(batch['sci_domain_ids']==3)])

    
    return batch
