# PoisonedRAG

Official repo of [USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25) paper: [PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models](https://arxiv.org/abs/2402.07867).

The first knowledge database corruption attack against Retrievals-Augmented Generation (RAG) system.

## 🚀 News
🎉 Jun 20, 2024: **PoisonedRAG gets accepted to [USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25)!**

🔥 Apr 20, 2024: **If you have any question or need other code or data, feel free to open an issue or email us!**

![Illustration of PoisonedRAG](PoisonedRAG.png "Illustration of PoisonedRAG")

## 🔍 Quick Usage

### 📃 Setup environment

```bash
conda create -n PoisonedRAG python=3.10
```
```bash
conda activate PoisonedRAG
```
```bash
pip install beir openai google-generativeai
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade charset-normalizer
pip3 install "fschat[model_worker,webui]"
```

### 💽 Dataset (optional, suggested)

When running our code, the datasets will be automatically downloaded and saved in `datasets`. You could also run this line to manually download datasets.

```bash
python prepare_dataset.py
```

### 🔑 Set API key

If you want to use PaLM 2, GPT-3.5, GPT-4 or LLaMA-2, please enter your api key in **model_configs** folder.

For LLaMA-2, the api key is your **HuggingFace Access Tokens**. You could visit [LLaMA-2's HuggingFace Page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) first if you don't have the access token.

Here is an example:

```json
"api_key_info":{
    "api_keys":[
        "Your api key here"
    ],
    "api_key_use": 0
},
```

### 📝 Reproduce our results
 
There are some hyperparameters in **run.py** such as LLMs and datasets:

**Note:** Currently we provide default setting for main results in our [paper](https://arxiv.org/abs/2402.07867). We will update and complete other settings later.

```python
test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",            # nq, hotpotqa, msmarco
    'split': "test",
    'query_results_dir': 'main',

    # LLM setting
    'model_name': 'palm2',           # palm2, gpt3.5, gpt4, llama(7b|13b), vicuna(7b|13b|33b)
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 0,

    # attack
    'attack_method': 'LM_targeted',  # LM_targeted (black-box), hotflip (white-box)
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times': 10,
    'M': 10,
    'seed': 12,

    'note': None
}
```

Execute **run.py** to reproduce experiments.

```bash
python run.py
```

### 🐱 Your own dataset
If you want to perform experiments on your own dataset, you could refer to **evaluate_beir.py**, prepare your dataset as the beir format and use this file to compute the retrieval scores.

## Acknowledgement

* Our code used the implementation of [corpus-poisoning](https://github.com/princeton-nlp/corpus-poisoning).
* The model part of our code is from [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection).
* Our code used [beir](https://github.com/beir-cellar/beir) benchmark.
* Our code used [contriever](https://github.com/facebookresearch/contriever) for retrieval augmented generation (RAG).



## Citation

If you use this code, please cite the following [paper](https://arxiv.org/abs/2402.07867):

```bib
@inproceedings{zou2025poisonedrag,
  title={$\{$PoisonedRAG$\}$: Knowledge corruption attacks to $\{$Retrieval-Augmented$\}$ generation of large language models},
  author={Zou, Wei and Geng, Runpeng and Wang, Binghui and Jia, Jinyuan},
  booktitle={34th USENIX Security Symposium (USENIX Security 25)},
  pages={3827--3844},
  year={2025}
}
```

