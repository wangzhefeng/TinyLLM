# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tests.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-13
# * Version     : 0.1.021300
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.gpt2_124M import main

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


expected = """
==================================================
                      IN
==================================================

Input text: Hello, I am
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])


==================================================
                      OUT
==================================================

Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]])
Output length: 14
Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous
"""




# 测试代码 main 函数
def test_main(capsys):
    main()
    captured = capsys.readouterr()
    
    # Normalize line endings and strip trailing whitespace from each line
    normalized_expected = "\n".join(line.rstrip() for line in expected.splitlines())
    normalized_output = '\n'.join(line.rstrip() for line in captured.out.splitlines())

    # Compare normalized strings
    assert normalized_output == normalized_expected

if __name__ == "__main__":
    test_main()
