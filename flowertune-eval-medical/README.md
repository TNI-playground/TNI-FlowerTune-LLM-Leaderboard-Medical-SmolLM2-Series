# Evaluation for Medical challenge

We build up a medical question answering (QA) pipeline to evaluate our fined-tuned LLMs.
Four datasets have been selected for this evaluation: [PubMedQA](https://huggingface.co/datasets/bigbio/pubmed_qa), [MedMCQA](https://huggingface.co/datasets/medmcqa), [MedQA](https://huggingface.co/datasets/bigbio/med_qa) and [CareQA](https://huggingface.co/datasets/HPAI-BSC/CareQA)


## Environment Setup

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/benchmarks/flowertune-llm/evaluation/medical ./flowertune-eval-medical && rm -rf flower && cd flowertune-eval-medical
```

Create a new Python environment (we recommend Python 3.11), activate it, then install dependencies with:

```shell
# From a new python environment, run:
pip install -r requirements.txt

# Log in HuggingFace account
huggingface-cli login
```

## Generate model decision & calculate accuracy

> [!NOTE]
> Please ensure that you use `quantization=4` to run the evaluation if you wish to participate in the LLM Leaderboard.

```
python eval.py \
--base-model-name-path=your-base-model-name \ # e.g., mistralai/Mistral-7B-v0.3
--peft-path=/path/to/fine-tuned-peft-model-dir/ \ # e.g., ./peft_1
--run-name=fl  \ # specified name for this run  
--batch-size=16 \
--quantization=4 \
--datasets=pubmedqa,medmcqa,medqa,careqa
```

Download fine-tuned model from [Google Drive](), put everything under `results` directory

```bash
cd /your_project_path/NI-FlowerTune-LLM-Leaderboard-Medical-SmolLM2-Series
bash script.sh
```

### Benchmark

| Challenges                       | pubmedqa   |   medqa    |  medmcqa    |   careqa      |  Avg       |
| :--------:                       | :--------: | :--------: | :--------:  | :--------:    | :--------: |
|SmolLM2-135M-Instruct (200Rounds) |    54.20   |   0.09     |   6.93      |    6.86       |  17.02     |
|SmolLM2-135M (200Rounds)          |    7.20    |   2.67     |   16.80     |    18.51      |  11.29     |
|SmolLM2-360M-Instruct (200Rounds) |   22.00    |   1.33     |   7.14      |    7.04       |   9.37     |
|SmolLM2-360M (200Rounds)          |   0.15     |   0.15     |   0         |    0.01       |   0.07     |


The model answers and accuracy values will be saved to `benchmarks/generation_{dataset_name}_{run_name}.jsonl` and `benchmarks/acc_{dataset_name}_{run_name}.txt`, respectively.


> [!NOTE]
> Please ensure that you provide all **four accuracy values (PubMedQA, MedMCQA, MedQA, CareQA)** for four evaluation datasets when submitting to the LLM Leaderboard (see the [`Make Submission`](https://github.com/adap/flower/tree/main/benchmarks/flowertune-llm/evaluation#make-submission-on-flowertune-llm-leaderboard) section).
