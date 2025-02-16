# SumEval4ALL: A Reference-Free Quality Evaluation Framework for Document Summarization using Large Language Models

## Environment Setup

Before running any evaluation, ensure you have installed the required dependencies. Since auto-evaluation and llm evaluation use different environments, install the respective requirements using:

For auto-evaluation:
```bash
pip install -r auto_eval_requirements.txt
```
For llm evaluation:
```bash
pip install -r llm_eval_requirements.txt
```


## Datasets Used
The following seven datasets are used for evaluation: **Arxiv**, **GovReportData**, **QAGS_CNN_DM**, **QAGS_X_SUM**, **PatentSumEval**, **SummEval**, and **TLDR**. 
All datasets are available in their raw form in the `raw_input` folder. To prepare them for evaluation, run the `preprocess_data.py` script. This will process the raw data and save the preprocessed versions in the `input_data` folder, making them ready for use.

## Conventional Evaluation Metrics
The following evaluation metrics have been used to assess the quality of summaries: `rouge_score`, `bert_score`, `summa_c_scores`, `summa_qa_scores`, `chrf_score`, `meteor_score`, `blanc_score`, `rouge_we_score`, `supert_score`, `bart_score`, `mover_score`, `qa_eval`, `quest_eval`, `feqa`, `bleu_score`, `fre_score`, and `dcr_score`.

### Running the Evaluation Metrics
To compute evaluation scores for each summary, open the `main_auto_evaluation.py` script and select `auto-evaluation` as the `tasks_to_perform`. Then execute:
```bash
python main_auto_evaluation.py
```

## Running Correlation for Evaluation Metrics
To compute correlation between evaluation metrics and human scores, modify the `main_auto_evaluation.py` script and select `correlation` as the `tasks_to_perform` to enable correlation computation and then run:
```bash
python main_auto_evaluation.py
```
This will generate correlation scores for each metric with human evaluations.

## Multi-Agent Evaluation
LLM evaluation can be performed using the `main_llm_evaluation.py` script. Define the `evaluator_model_ids` and `leader_model_ids` in the script along with other configurations and select the tasks from `generate-summary` `evaluate-summary` or `improve-evaluate-summary` before running:
```bash
python main_llm_evaluation.py
```
After evaluation, use the `averaging_score.py` script in the `multi_agent` folder to aggregate results based on the required evaluators:
```bash
python multi_agent/averaging_score.py
```

## Running Correlation for Multi-Agent Evaluation
Perform correlation analysis on aggregated evaluation results by modifying and executing the `prepare_correlation_multi_agent.py` script available in the `multi_agent` folder:
```bash
python multi_agent/correlation_analysis.py
```
This script will run correlation analysis for each of the provided datasets.

---
This README provides instructions to preprocess datasets, run evaluation metrics, compute correlations, and perform multi-agent evaluation. Ensure all dependencies are installed before running the scripts.

