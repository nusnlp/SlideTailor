
<p align="center">
  <h1 align="center">SlideTailor: Personalized Presentation Slide Generation for Scientific Papers</h1>

  <p align="center">
    <a href="https://wenzhengzeng.github.io/">Wenzheng Zeng*</a>,
<!--     · -->
    <a href="https://scholar.google.com/citations?user=GoYztjwAAAAJ&hl=en">Mingyu Ouyang*</a>,
<!--     · -->
    <a href="https://openreview.net/profile?id=~Langyuan_Cui1">Langyuan Cui*</a>,
<!--     · -->
    <a href="https://scholar.google.com.tw/citations?user=FABZCeAAAAAJ">Hwee Tou Ng†</a>,
    
  </p>
  <p align="center">National University of Singapore</p>
  <h3 align="center">AAAI 2026</h3>

  <h3 align="center"> 
  <a href="https://arxiv.org/abs/2512.20292">📄 Paper</a> &nbsp; | &nbsp;
    <a href="https://huggingface.co/papers/2512.20292">🤗 Daily paper</a> &nbsp; | &nbsp;
  <a href="https://huggingface.co/datasets/yyyang/SlideTailor-PSP-dataset">🤗 Dataset</a> &nbsp; | &nbsp;
  <a href="https://drive.google.com/drive/folders/1N8p1A4eW8Nrrc2fN5NnIutG0og9u_GIy?usp=sharing">🖼️ Poster</a> &nbsp; | &nbsp;
  <a href="https://www.youtube.com/watch?v=NT5kWE6j_Vw">▶️ Video</a> &nbsp; | &nbsp;
<a href="https://drive.google.com/drive/folders/1N8p1A4eW8Nrrc2fN5NnIutG0og9u_GIy?usp=sharing">▶️ Slides</a> &nbsp; | &nbsp;
<a href="https://x.com/alexzeng1206/status/2005535755003904071">▶️ X (Twitter)</a> &nbsp;
</a> </h3>


</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
    <img src="pic/fig2.png" width="90%"/>

</p>


</p>
This repository contains the official implementation of the AAAI 2026 paper "SlideTailor: Personalized Presentation Slide Generation for Scientific Papers".

## 🔆 Overview
We argue that presentation design is inherently subjective. Users have different preferences in terms of narrative structure, emphasis, conciseness, aesthetic choices, etc. 

So in this work, we ask: **Can we better model such diverse user preferences for personalized paper-to-slides generation?**

We make the following contributions:

- **Task:** We introduce and properly define a new task that conditions paper-to-slide generation on user-specified preferences.
- **System:** We propose a human behavior-inspired agentic framework, SlideTailor, that progressively generates editable slides in a user-aligned manner.
- **Evaluation:** We construct a benchmark dataset that captures diverse user preferences, with meticulously designed interpretable metrics for robust evaluation.
- **Open Source:** We release the source code and data to the community.


## 🔥 News
* [2025-11] Our work is accepted to AAAI 2026!
* [2025-11] We release our code and data to the community!

## 🛠️ Environment
1. Create a new conda environment
```sh
conda create -n slidetailor python=3.11
```
2. Install python dependency
```sh
pip install -r requirements.txt
```
3. Install other dependencies
```sh
#If using Ubuntu
sudo apt-get install libreoffice poppler-utils
```
4. Prepare your API key (named as api_key.json)

```
{
"openai_api_key": "sk-proj-1234"
}
```
## 📦 Dataset
Please prepare the data and the corresponding config files according to the instructions on the [PSP Dataset](https://huggingface.co/datasets/yyyang/SlideTailor-PSP-dataset) page.

## 🤖 Inference
Remember to modify the relevant paths in the following script before running it.

```
sh run.sh
```

## ⚖️ Evaluation
Remember to modify the relevant paths in the following script before running it.
```
sh eval.sh
```


## 🙏 Acknowledgments

We would like to thank [PPTAgent](https://github.com/icip-cas/PPTAgent) for their valuable contribution and for making their codebase available to the community.


## 📖 Citation

If you find this work helpful, please kindly cite our paper:

  ```
  @inproceedings{slidetailor,
    title={SlideTailor: Personalized Presentation Slide Generation for Scientific Papers},
    author={Zeng, Wenzheng and Ouyang, Mingyu and Cui, Langyuan and Ng, Hwee Tou},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2026}
  }
  ```