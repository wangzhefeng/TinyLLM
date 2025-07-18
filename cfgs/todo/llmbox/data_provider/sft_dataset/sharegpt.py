from .sftdataset import SFTDataset
import torch

class ShareGPTDataset(SFTDataset):
    """
    ShareGPT is an open-source Chrome Extension for you to share your ChatGPT conversations.
    And the dataset is scraped from that extension.
    """
    
    def process(self, tokenizer):
        input_ids = []
        labels = []
        list_data_dict = self.load_data()
        for example in list_data_dict:
            tmp1 = []
            tmp2 = []
            for s, t in zip(example['conversations'][::2], example['conversations'][1::2]):
                s = tokenizer.apply_chat_template([{'role':'user','content':s['value']}],tokenize=False)
                t = tokenizer.apply_chat_template([{'role':'assistant','content':t['value']}],tokenize=False)
                input_id, label = self.encode_src_tgt(s, t, tokenizer)
                tmp1.append(input_id)
                tmp2.append(label)
            input_ids.append(torch.cat(tmp1))
            labels.append(torch.cat(tmp2))
        return input_ids, labels
