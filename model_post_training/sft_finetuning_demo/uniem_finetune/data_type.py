# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_type.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-21
# * Version     : 0.1.102121
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings


from uniem.data_structures import (
    RecordType, 
    PairRecord, 
    TripletRecord, 
    ScoredPairRecord,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
# params
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
print(f'record_types: {[record_type.value for record_type in RecordType]}')


# 数据类型
pair_record = PairRecord(
    text = '肾结石如何治疗？', 
    text_pos = '如何治愈肾结石'
)
print(f'pair_record: {pair_record}')

triplet_record = TripletRecord(
    text = '肾结石如何治疗？', 
    text_pos = '如何治愈肾结石', 
    text_neg = '胆结石有哪些治疗方法？'
)
print(f'triplet_record: {triplet_record}')

# 1.0 代表相似，0.0 代表不相似
scored_pair_record1 = ScoredPairRecord(
    sentence1='肾结石如何治疗？', 
    sentence2='如何治愈肾结石', 
    label=1.0
)
scored_pair_record2 = ScoredPairRecord(
    sentence1='肾结石如何治疗？', 
    sentence2='胆结石有哪些治疗方法？', 
    label=0.0
)
print(f'scored_pair_record: {scored_pair_record1}')
print(f'scored_pair_record: {scored_pair_record2}')


# 2.0 代表相似，1.0 代表部分相似，0.0 代表不相似
scored_pair_record1 = ScoredPairRecord(
    sentence1 = '肾结石如何治疗？', 
    sentence2 = '如何治愈肾结石', 
    label = 2.0
)
scored_pair_record2 = ScoredPairRecord(
    sentence1 = '肾结石如何治疗？', 
    sentence2 = '胆结石有哪些治疗方法？', 
    label = 1.0
)
scored_pair_record3 = ScoredPairRecord(
    sentence1 = '肾结石如何治疗？', 
    sentence2 = '失眠如何治疗', 
    label = 0
)
print(f'scored_pair_record: {scored_pair_record1}')
print(f'scored_pair_record: {scored_pair_record2}')
print(f'scored_pair_record: {scored_pair_record3}')



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
