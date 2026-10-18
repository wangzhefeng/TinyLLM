#!/bin/bash

# 默认情况下, 当使用文件执行Python代码时, 
# Python解释器会自动将该文件的字节码版本写入磁盘.  
# 比如,  module.pyc. 这些“.pyc”文件不应该加入到您的源代码仓库. 
# 理论上, 出于性能原因, 此行为默认为开启.  没有这些字节码文件, 
# Python会在每次加载文件时 重新生成字节码. 


# 禁止在执行 .py 文件时生成 __pycache__/.pyc 文件
# vim ~/.profile
# export PYTHONDONTWRITEBYTECODE=1

# 删除全部 __pycache__/.pyc 文件
sudo find . -type f -name "*.py[co]" -delete -or -type d -name "__pycache__" -delete 

# ----------------------------------------------
# 不要在源代码仓库中加入 .pyc 文件
# vim .gitignore
# *.py[cod]     # 将匹配 .pyc、.pyo 和 .pyd文件
# __pycache__/  # 排除整个文件夹
