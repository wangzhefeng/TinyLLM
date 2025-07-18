
import os
import json
import random
from typing import Dict

import torch
from datasets import load_dataset,load_from_disk


class SFTDataset:
    """
    This is the base class for all SFT datasets.
    """

    IGNORE_INDEX = -100
    instruction_template = "\n### Instruction:\n"
    response_template = "\n### Output:\n"
    format_template = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
            "{input}" + response_template
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
            response_template 
        ),
    }
    
    def __init__(self, args, tokenizer):
        self.args = args
        self.block_size = self.args.model_max_length
        self.tokenizer = tokenizer
        data_path = args.data_path
        
        pth_file = f"{data_path}.pth"
        retokenize = False
        if Path(pth_file).exists():
            data_dict = torch.load(pth_file)
            prev_tokenizer_config = data_dict['tokenizer_config']
            current_tokenizer_config = str(tokenizer)
            if prev_tokenizer_config != current_tokenizer_config:
                retokenize = True
            else:
                print("Loading tokenized data from cache")
                self.input_ids = data_dict['input_ids']
                self.labels = data_dict['labels']
        else:
            retokenize = True

        if retokenize:
            self.input_ids, self.labels = self.process(self.tokenizer)
            if self.args.packing:
                self.input_ids, self.labels = self.packed_sft_examples(self.input_ids, self.labels)

            if torch.distributed.get_rank() == 0:
                checkpoint = {'input_ids': self.input_ids, 'labels': self.labels, 'tokenizer_config': str(tokenizer)}
                torch.save(checkpoint, pth_file)

        self.shuffle_index = list(range(len(self.input_ids)))
        random.shuffle(self.shuffle_index)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        index = self.shuffle_index[i]
        return dict(input_ids=self.input_ids[index], labels=self.labels[index])
    
    def encode_src_tgt(self, s, t, tokenizer):
        source_id = tokenizer.encode(s, max_length=tokenizer.model_max_length, truncation=True)
        tokenizer.add_eos_token = True
        input_id = tokenizer.encode(s + t, max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')[0]
        tokenizer.add_eos_token = False
        label = input_id.clone()
        label[:len(source_id)] = self.IGNORE_INDEX
        
        return input_id, label
    
    def packed_sft_examples(self, input_ids, labels):
        new_input_ids, new_labels = [torch.tensor([], dtype=input_ids[0].dtype)], [torch.tensor([], dtype=input_ids[0].dtype)]
        lengths = [[]]
        for input_id, label in zip(input_ids, labels):
            if len(new_input_ids[-1]) + len(input_id) <= self.block_size:
                new_input_ids[-1] = torch.cat((new_input_ids[-1], input_id))
                new_labels[-1] = torch.cat((new_labels[-1], label))
                lengths[-1].append(len(input_id))
            else:
                new_input_ids.append(input_id)
                new_labels.append(label)
                lengths.append([len(input_id)])

        return new_input_ids, new_labels

    def load_data(self):
        """
        Load data.
        """
        data_path = self.args.data_path
        
        if data_path.endswith('.jsonl'):
            list_data_dict = [
                json.loads(l.strip()) 
                for l in open(data_path, encoding='utf-8')
            ]
        elif data_path.endswith('.json'):
            try: # if it's really json format
                list_data_dict = json.load(open(data_path, encoding='utf-8'))
            except: # if it's a list of json
                list_data_dict = [json.loads(l.strip()) for l in open(data_path, encoding='utf-8')]
        elif os.path.isdir(data_path):
            list_data_dict = load_from_disk(data_path)['train']
        else: 
            try:
                list_data_dict = load_dataset(data_path)['train']
            except:
                raise ValueError(f"Unsupported file format: {data_path}") # TODO: Add support for other file formats
        
        return list_data_dict
        
    def process(self,tokenizer):
        """Process the dataset and return input_ids and labels."""
        input_ids = []
        labels = []
        list_data_dict = self.load_data()

        for example in list_data_dict:
            example['response'] = example.pop('output') # change the key name from 'output' to 'response'
            s = (self.format_template["prompt_input"].format_map(example) if 'input' in example.keys() else self.format_template["prompt_no_input"].format_map(example)).strip()
            t = example['response'].strip()
            input_id, label = self.encode_src_tgt(s, t, tokenizer)
            input_ids.append(input_id)
            labels.append(label)
        return input_ids, labels
    

