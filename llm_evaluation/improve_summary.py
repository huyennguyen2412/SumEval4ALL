### Use this code to improve and score on the summaries

import os
import pandas as pd
from tqdm import tqdm
import sys
# sys.path.append('/home/lavanya/lavanya/Sum_Eval')

# from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import openpyxl
import ast
import pandas as pd
import transformers
from transformers import pipeline
# import my_tokens

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
# import utils

# os.environ['HF_TOKEN'] = my_tokens.my_hf_token
load_dotenv()
access_token = os.getenv("ACCESS_TOKEN")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def improve_summary_openllm(data_input, summary2improve, eval_feedback, model, tokenizer, max_new_tokens=200):
    preprompt = f"""I would like you to improve the provided summary based on the evaluation feedback given to that summary. 
    Refer to the original document to make improvements. The output must be only the improved summary, nothing else.""" #

    llm_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation", #'text-classification', "text-generation"
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
        truncation = True
    )
   
    prompt_in_chat_format = [
    {
        "role": "system",
        "content": preprompt },
    {
        "role": "user",
        "content": f"""Here is the summary that needs to be improved: {summary2improve}\n\n 
        Here is the quality evaluation feedback given to that summary: {eval_feedback}\n\n
        Here is the original document that needs to be summarized: {data_input}"""
        }

        ]
    
    llm_prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    summary = llm_pipeline(llm_prompt_template)[0]["generated_text"]

    print('completed one row...')
    
    return summary


def main_improving(model_name: str):
    """"
    model_name: llama3, gpt3.5
    """
    in_file_paths = [r'arxiv_pubmed_output/summary_test_arxiv_1000.xlsx_updated_eval_scores.xlsx',r'arxiv_pubmed_output/summary_test_pubmed_1000.xlsx_updated_eval_scores.xlsx']

    # in_file_path = r'/home/lavanya/lavanya/Sum_Eval/new_dataset_outputs/optimalPrompt_CNN_updated_eval_scores.csv' 
    # with open (in_file_path, 'r') as f:
    #     data = pd.read_csv(f)

    model = AutoModelForCausalLM.from_pretrained(model_name,token=access_token, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token, model_max_length=8500,truncation = True)


    for path in in_file_paths:
        
        out_file_path = os.path.join(path.split('/')[0], path.split('/')[1].split('.')[0]+'_final_summary.xlsx')
        data = pd.read_excel(path)
        # data = data[:10]
            
        print(data.columns)

        # improved_summaries = []
        for i, row in data.iterrows():
            if row['meta-l_overall'] < 4:
                improved_summary = improve_summary_openllm (data_input=row['article'],
                                                                        summary2improve=row['gen_summary'],
                                                                        eval_feedback= row[f'meta-l_raw_response'],
                                                                        model = model,
                                                                        tokenizer=tokenizer,
                                                                        max_new_tokens=200,
                                                                        )
                # improved_summaries.append(improved_summary)
                data.loc[i, f'{model_name}_improved'] = improved_summary
            else:
                data.loc[i, f'{model_name}_improved'] = row['gen_summary']

            data.to_excel(out_file_path, index=False)
        
        print('done for one file')

        # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" #"meta-llama/Llama-2-7b-chat-hf" 
        # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, local_files_only = True)
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # if model_name == 'llama3':
    #     model_id = "meta-llama/Meta-Llama-3-8B-Instruct" #"meta-llama/Llama-2-7b-chat-hf" 
    #     model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, local_files_only = True)
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
        #feedback_path = r'/home/lavanya/lavanya/Sum_Eval/optimal_CNN_sample.csv' 
        #eval_results = pd.read_csv(feedback_path)
        #raw_outputs = eval_results[f"meta-l_raw_response"].str.replace("```python", '')#.replace("```", '')
        #raw_outputs = raw_outputs.str.replace("```", '')
    
    # for i, raw in enumerate(list(raw_outputs)):
    #     print(i, '-->', raw)
    #     print(ast.literal_eval(raw.strip()))
    #     print()

    #eval_outputs = [ast.literal_eval(raw.strip()) for raw in raw_outputs]
    #print(eval_outputs)
    # feedbacks = [out['explanation'] for out in eval_outputs]
    #if list(data['patent_id']) == list(eval_results['patent_id']):
        #data[f'{model_name}_feedback'] = eval_outputs
    

    
    # # data = data[:3]
    # if model_name in ['llama3', 'llama2']:
    #     # data[f'{model_name}_improved'] = data.apply(lambda row: improve_summary_openllm (data_input=row['Abstract'] + "\n" + row['Claims'],
    #     #                                                                 summary2improve=row[model_id.split('/')[1]],
    #     #                                                                 eval_feedback= row[f'{model_name}_feedback'],
    #     #                                                                 model = model,
    #     #                                                                 tokenizer=tokenizer,
    #     #                                                                 max_new_tokens=200,
    #     #                                                                 ), axis = 1)
    #     improved_summaries = []
    #     for i, row in data.iterrows():
    #         #print(type(row[f'{model_name}_feedback']))
    #         # print(row[f'{model_name}_feedback'])
    #         feedback_dict = row[f'meta-l_raw_response']
    #         if row['meta-l_overall'] < 4:
    #             improved_summary = improve_summary_openllm (data_input=row['article'],
    #                                                                     summary2improve=row['gen_summary'],
    #                                                                     eval_feedback= row[f'meta-l_raw_response'],
    #                                                                     model = model,
    #                                                                     tokenizer=tokenizer,
    #                                                                     max_new_tokens=200,
    #                                                                     )
    #             improved_summaries.append(improved_summary)
    #         else:
    #             improved_summaries.append(row['gen_summary'])
            
    #     data[f'{model_name}_improved'] = improved_summaries
      
    # with open (f"{in_file_path.split('.')[0]}_improved_{model_name}_CNN_Improvised.csv", 'w', encoding = 'utf-8', newline = '') as f:
    #     data.to_csv(f)


if __name__ == '__main__':
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    main_improving(model_name)