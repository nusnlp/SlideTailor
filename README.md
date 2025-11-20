
<p align="center">
  <h1 align="center">SlideTailor: Personalized Presentation Slide Generation for Scientific Papers</h1>
  <p align="center">


  <h3 align="center">AAAI 2026</h3>


</a> </h3>
  <div align="center"></div>
</p>


</p>
This repository contains the official implementation of the AAAI 2026 paper "SlideTailor: Personalized Presentation Slide Generation for Scientific Papers".


## 🔆 Highlights 

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
2. Install dependency
```sh
pip install -r requirements.txt
```
3. Prepare your API key (named as api_keys.json)

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