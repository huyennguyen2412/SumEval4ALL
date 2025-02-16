import pandas as pd 
import os
from llm_evaluation.llm_evaluation_metrices import metrices_dict

input_file_paths = [
                    # r'output_multi_agent_new_models/output_llm_evaluation/arxiv_data.xlsx', 
                    # r'output_multi_agent_new_models/output_llm_evaluation/gov_report_data.xlsx', 
                    r'output_multi_agent_new_models/output_llm_evaluation/patent_sum_eval_data.xlsx', 
                    # r'output_multi_agent_new_models/output_llm_evaluation/qags_cnn_dm_data.csv', 
                    # r'output_multi_agent_new_models/output_llm_evaluation/qags_x_sum_data.csv', 
                    r'output_multi_agent_new_models/output_llm_evaluation/summ_eval_data.csv', 
                    # r'output_multi_agent_new_models/output_llm_evaluation/tldr_data.xlsx'
                    ]

for input_file_path in input_file_paths:
    if input_file_path.split('.')[-1] == 'xlsx':
        data = pd.read_excel(input_file_path)
    elif input_file_path.split('.')[-1] == 'csv':
        data = pd.read_csv(input_file_path)

    model_ids = ["meta-llama/Meta-Llama-3.1-8B-Instruct","Saxo/Linkbricks-Horizon-AI-Avengers-V6-32B","microsoft/phi-4","Qwen/Qwen2-7B-Instruct"]
    # model_ids = ["microsoft/phi-4","Qwen/Qwen2-7B-Instruct"]
    summary_columns = [col for col in data.columns if ((col.endswith('summary')) and not (col.startswith('llm_')))]
    metrices = metrices_dict.get(input_file_path.split('/')[-1].split('.')[0])

    aggregate_metric_dict = {f'{summary}_{metric}' : [f'{summary}_{metric}_{model}' for model in model_ids] for metric in metrices for summary in summary_columns }

    for metric, columns in aggregate_metric_dict.items():
        for summary_model in summary_columns:
            data[f'average_{metric}_{summary_model}'] = data[columns].mean(axis=1)
            data[f'mode_{metric}_{summary_model}'] = data[columns].mode(axis=1)[0] 

    output_path = os.path.join('output_auto_evaluation_multi_agent_new_models',os.path.split(input_file_path)[-1].split('.')[0],os.path.split(input_file_path)[-1])

    if input_file_path.split('.')[-1] == 'xlsx':
        data.to_excel(output_path)
    elif input_file_path.split('.')[-1] == 'csv':
        data.to_csv(output_path)


    



