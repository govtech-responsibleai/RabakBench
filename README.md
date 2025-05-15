# RabakBench: Scaling Human Annotations to Construct Localized Multilingual Safety Benchmarks for Low-Resource Languages

RabakBench is a multilingual safety benchmark designed for Singapore's linguistic content. It supports evaluation of popular open-source and closed-source content moderation systems in low-resource and culturally diverse languages, covering Singlish (an English-based creole) and local variants of Chinese, Malay, and Tamil. These languages are often underrepresented in existing benchmarks, posing challenges for large language models (LLMs) and their safety classifiers.

By releasing RabakBench, we aim to advance the study of AI safety in low-resource languages by enabling robust safety evaluation in multilingual settings and providing a reproducible framework for building localized safety datasets.

> [!TIP]
> Explore the dataset through this [Jupyter Notebook](dataset_eda.ipynb)

## Dataset Construction

RabakBench comprises over 5,000 examples across six harm categories with severity levels. The dataset was constructed through a scalable three-stage pipeline:

1. Generate: Adversarial example generation by augmenting real Singlish web content with LLM-driven red teaming
2. Label: Semi-automated multi-label safety annotation using majority-voted LLM labelers aligned with human judgments
3. Translate: High-fidelity translation preserving linguistic nuance and toxicity across languages.

<div align="center">
<img src="assets/rabakbench_pipeline.png" alt="RabakBench Construction Pipeline" style="width:75%"/>
</div>

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

Due to sensitive content, access to the dataset is provided through a gated process:

- For reviewers: Private access is granted via Kaggle. Please refer to the OpenReview submission for the link
- For researchers: We plan to support controlled access in the future with terms of use and intent verification, to ensure responsible usage aligned with our goals of improving multilingual AI safety. Further details will be made available soon.

Download the 4 datasets and place them in the appropriate folders under the `/data` directory:

- `rabakbench_en.csv → /data/en/rabakbench_en.csv`
- `rabakbench_ms.csv → /data/ms/rabakbench_ms.csv`
- `rabakbench_ta.csv → /data/ta/abakbench_ta.csv`
- `rabakbench_zh.csv → /data/zh/rabakbench_zh.csv`

### 3. Configure API keys

Create a .env file and add the following API keys:

```bash
# OpenAI Moderation API
export OPENAI_API_KEY=XXXXXXXXXX

# Azure Content Safety API
export AZURE_CONTENT_SAFETY_ENDPOINT="https://XXX.azure.com/"
export AZURE_CONTENT_SAFETY_KEY=XXXXXXXXXX

# AWS Moderation API
export AWS_ACCESS_KEY_ID=XXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=XXXXXXXXXX
export AWS_SESSION_TOKEN=XXXXXXXXXX

# GCP Project ID (Model Armor and Perspective API)
export GCP_PROJECT_ID=XXXXXXXXXX

# Perspective API
export PERSPECTIVE_API_KEY=XXXXXXXXXX

# Fireworks API (For LlamaGuard and select open-sourced models)
export FIREWORKS_API_KEY=XXXXXXXXXX
```

### 4. Hugging Face Authentication (for open-source models)

Some open-source models used in our benchmark (e.g., DuoGuard, WildGuard) are loaded via the 🤗 transformers package. To run these models, you'll need to authenticate with Hugging Face:

```bash
huggingface-cli login
```

Make sure you have an access token from https://huggingface.co/settings/tokens.

## Evaluation

### 1. Run content moderators on RabakBench

```bash
python moderate.py
```

### 2. Evaluate results

```bash
python evaluate.py
```

## Results

Evaluations of 11 prominent open-source and closed-source guardrail classifiers revealed significant performance degradation on this localized, multilingual benchmark. More details on the evaluation setup can be found in our paper.

Refer to ./dataset_eda.ipynb and the ./results folder for the full set of evaluation metrics, per-language scores, and error breakdowns.

| Guardrail                | Singlish | Chinese | Malay | Tamil | Average |
| :----------------------- | :------: | :-----: | :---: | :---: | :-----: |
| AWS Bedrock Guardrail    |  66.50   |  0.06   | 17.47 | 0.06  |  21.28  |
| Azure AI Content Safety  |  66.70   |  73.62  | 66.18 | 53.86 |  65.09  |
| Google Cloud Model Armor |  62.37   |  67.95  | 71.26 | 73.56 |  68.78  |
| OpenAI Moderation        |  66.00   |  68.20  | 59.00 | 0.69  |  50.01  |
| Perspective API          |  37.80   |  50.46  | 18.60 | 0.10  |  26.97  |
| DuoGuard 0.5B            |  42.28   |  58.15  | 31.70 | 43.55 |  43.92  |
| LlamaGuard 3 8B          |  54.76   |  53.05  | 47.05 | 46.84 |  50.42  |
| LlamaGuard 4 12B         |  60.53   |  54.20  | 62.36 | 73.77 |  62.72  |
| PolyGuard 0.5B           |  67.51   |  75.70  | 58.00 | 21.27 |  55.62  |
| ShieldGemma 9B           |  41.37   |  31.85  | 29.23 | 22.78 |  31.31  |
| WildGuard 7B             |  78.89   |  68.82  | 35.77 | 0.23  |  44.45  |

## Citations

Please cite our paper if you find RabakBench helpful in your research!
