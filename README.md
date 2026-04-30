# 🏗️ **AssemLM: Spatial Reasoning Multimodal Large Language Models for Robotic Assembly**

<div align="center">

[Zhi Jing](https://scholar.google.com/citations?user=dlbY-nkAAAAJ&hl=zh-CN)<sup>1,2</sup>, [Jinbin Qiao](https://scholar.google.com/citations?user=L3_Wl_0AAAAJ&hl=zh-CN)<sup>2,3</sup>, [Ouyang Lu](https://scholar.google.com/citations?user=QXCaKP4AAAAJ&hl=zh-CN)<sup>2,4</sup>, [Jicong Ao](https://scholar.google.com/citations?user=AA4WipQAAAAJ&hl=en)<sup>2</sup>, [Shuang Qiu](https://www.cityu.edu.hk/stfprofile/shuanqiu.htm)<sup>5</sup>, [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en)<sup>1,\*</sup>, [Chenjia Bai](https://baichenjia.github.io/)<sup>2,\*</sup>

<sup>1</sup>Fudan University<sup>†</sup>, <sup>2</sup>Institute of Artificial Intelligence (TeleAI), China Telecom<sup>†</sup>,

<sup>3</sup>Tianjin University, <sup>4</sup>Northwestern Polytechnical University, <sup>5</sup>City University of Hong Kong

<sup>\*</sup> Equal advising | <sup>†</sup> Equally leading organizations

<p align="center">
  <a href="https://arxiv.org/pdf/2604.08983"><img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2604.08983"><img src="https://img.shields.io/badge/arXiv-2604.08983-b31b1b" alt="arXiv"></a>
  <a href="https://huggingface.co/TeleEmbodied/AssemLM-V1/tree/main"><img src="https://img.shields.io/badge/Model-HuggingFace-yellow" alt="Model"></a>
  <a href="https://huggingface.co/datasets/TeleEmbodied/AssemLM/tree/main"><img src="https://img.shields.io/badge/Datasets-HuggingFace-yellow" alt="Datasets"></a>
  <a href="https://assemlmhome.github.io/"><img src="https://img.shields.io/badge/Project%20Page-Home-green" alt="Project Page"></a>
  <a href="https://github.com/TeleHuman/AssemLM"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="Code"></a>
</p>

</div>

![](assets/main.png)

## 🚀 News

- **[2026-04-29]** 🔓 Open-source the inference code, AssemLM-V1 weights, and demo dataset for inference.
- **[2026-04-16]** 🗺️ Announce the open-source plan.
- **[2026-04-10]** 📄 Upload the paper to arXiv: [paper](https://arxiv.org/abs/2604.08983)
- **[2026-03-15]** 🎉 Release the first version of the [project page](https://assemlmhome.github.io/).
- **[2026-03-05]** 🏗️ Create the [project page](https://assemlmhome.github.io/) and [code repository](https://github.com/TeleHuman/AssemLM).

## ⚙️ Setup Environment

### Installation Steps

#### 1. **Clone the repository**

```sh
git clone https://github.com/TeleHuman/AssemLM.git
cd AssemLM
```

#### 2. **Create & Build conda env**

```sh
conda create -n assemlm python=3.10.14 -y
conda activate assemlm
bash setting.sh
```

## 🗺️ Open-Source Plan

- [x] 🔓 Release **AssemLM-V1 weights**, **inference code**, and a **demo dataset**.
- [ ] 📦 Release the **majority of the AssemBench dataset**.
- [ ] 📚 Release **additional datasets and benchmark resources**.
- [ ] 🧠 Release the **training code**.
- [ ] ⚙️ Release the **data processing pipeline**.
- [ ] 🚀 Release **updated and improved model weights**.
