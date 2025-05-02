
# 数据集下载

## 安装 ModelScope

```bash
$ pip install modelscope
```

## 命令行下载

下载完整数据集

```bash
$ modelscope download --dataset gongjy/minimind_dataset
```

下载单个文件到指定本地文件夹（以下载 `README.md` 到当前路径下 `dir` 目录为例）

```bash
$ modelscope download --dataset gongjy/minimind_dataset README.md --local_dir ./dir
```

## SDK 下载

```python
#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('gongjy/minimind_dataset', subset_name='default', split='train')
#您可按需配置 subset_name、split，参照“快速使用”示例代码
```

## Git 下载

请确保 lfs 已经被正确安装

```bash
$ git lfs install
```

```bash
$ git clone https://www.modelscope.cn/datasets/gongjy/minimind_dataset.git
```
