<details><summary>目录</summary><p>

- [nanoVLM](#nanovlm)
    - [项目介绍](#项目介绍)
    - [什么是视觉语言模型](#什么是视觉语言模型)
- [参考](#参考)
</p></details><p></p>

# nanoVLM

## 项目介绍

* `models/vision_transformer.py`
    -  视觉主干网络(标准的视觉 Transformer): Google SigLIP 视觉编码器(google/siglip-base-patch16-224)
* `models/language_model.py`
    - 语言主干网络：Llama 3(HuggingFaceTB/SmolLM2-135M)
* `models/modality_projector.py`
    - 模态投影模块：将视觉和文本模态进行对齐

## 什么是视觉语言模型






# 参考

* [nanoVLM: The simplest repository to train your VLM in pure PyTorch](https://huggingface.co/blog/zh/nanovlm)
* [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM)
