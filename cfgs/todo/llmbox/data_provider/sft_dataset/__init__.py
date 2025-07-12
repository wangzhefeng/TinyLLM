from collections import OrderedDict

import torch

from .alpaca import AlpacaDataset
from .belle import BelleDataset
from .dolly import DollyDataset
from .evol_instruct import EvolInstructDataset
from .flan import FlanDataset
from .lima import LimaDataset
from .openassistant import OpenAssistantDataset
from .self_instruct import SelfInstructDataset
from .sharegpt import ShareGPTDataset


# You can add your own dataset name and corresponding class here
DATASETNAMEMAP = OrderedDict({
    "alpaca": AlpacaDataset,
    "belle": BelleDataset,
    "self_instruct": SelfInstructDataset,
    "evol_instruct": EvolInstructDataset,
    "dolly": DollyDataset,
    "lima": LimaDataset,
    "sharegpt": ShareGPTDataset,
    "openassistant": OpenAssistantDataset,
    "flan": FlanDataset,
})


class SFTDataset:
    
    def __new__(self, args, tokenizer):
        datapath = args.data_path
        for datasetname, datasetclass in DATASETNAMEMAP.items():
            # if the datasetname in the datapath, then we select this dataset
            if datasetname in datapath:
                if torch.distributed.get_rank() == 0:
                    print(f"Using {datasetname} dataset class")
                
                return datasetclass(args, tokenizer)

        # failed to find the dataset
        raise ValueError(
            f"Your {datapath} should contain names like these: {DATASETNAMEMAP.keys()}, so that it can find our sftdataset class. Or you can add your own dataset class."
        )
