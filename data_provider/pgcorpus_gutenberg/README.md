<details><summary>目录</summary><p>

- [数据下载](#数据下载)
- [数据处理](#数据处理)
- [预训练](#预训练)
- [参考](#参考)
</p></details><p></p>


# 数据下载

```bash
$ git clone https://github.com/pgcorpus/gutenberg.git
```

```bash
$ cd gutenberg
```

```bash
$ pip install -r requirements.txt
```

```bash
$ python get_data.py
```

# 数据处理

* 数据处理

```bash
$ python process_data.py
```

* 准备数据集

```bash
$ python prepare_dataset.py \
$   --data_dir gutenberg/data/raw \
$   --max_size_mb 500 \
$   --output_dir gutenberg_preprocessed
```

# 预训练

```bash
$ python pretraining_simple.py \
$   --data_dir "gutenberg_preprocessed" \
$   --n_epochs 1 \
$   --batch_size 4 \
$   --output_dir model_checkpoints
```

# 参考

* [pgcorpus/gutenberg](https://github.com/pgcorpus/gutenberg)
* [pretraining_on_gutenberg](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/03_bonus_pretraining_on_gutenberg)
