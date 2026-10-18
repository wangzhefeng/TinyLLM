# -*- coding: utf-8 -*-

# ***************************************************
# * File        : vocab_download.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-08-03
# * Version     : 1.0.080319
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import requests
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def download_vocab(model_file_path, model_name="117M"):
    # model file downloaded path
    model_file_path.mkdir(parents=True, exist_ok=True)
    # model file download
    for filename in ['encoder.json', 'vocab.bpe']:
        file = model_file_path.joinpath(filename)
        if not file.exists():
            # download
            r = requests.get(
                f"https://openaipublic.blob.core.windows.net/gpt-2/models/{model_name}/{filename}", 
                stream=True
            )
            # save
            with open(file, 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
