import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from torch.nn import DataParallel
import ast  # For parsing dictionary from model responses
import re  # For extracting dictionary-like patterns
from dotenv import load_dotenv
import nltk
import numpy as np
from llm_evaluation.scores_extract import extract_scores_if_missing
from llm_evaluation.llm_evaluation_prompts import prompt_dict
from llm_evaluation.llm_config import llm_config_dict
from llm_evaluation.llm_evaluation_metrices import metrices_dict


class EvaluateSummary:
    def __init__(self, input_file_paths, article_column, summary_columns: list, 
                 summary_sentence_column = None ,levels = 'summary-level',
                 is_improved = False,model_ids=["meta-llama/Meta-Llama-3.1-8B-Instruct"], 
                 output_dir: str = ''):
        load_dotenv()
        nltk.download('punkt')
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.input_file_paths = input_file_paths
        self.output_dir = output_dir
        self.levels = levels
        self.is_improved = is_improved
        self.model_ids = model_ids
        self.article_column = article_column
        self.summary_columns = summary_columns
        self.summary_sentence_column = summary_sentence_column
        self.task = 'evaluate-summary' if not is_improved else 'improve-evaluate-summary' 
        

    def evaluate_summary_openllm(self, pre_prompt, data_input, summary, model, tokenizer):
        """Generate model response using the pipeline."""
        try:
            if isinstance(model, DataParallel):
                model = model.module

            llm_pipeline = pipeline(
                model=model,
                tokenizer=tokenizer,
                task = llm_config_dict[self.task]['task'],
                batch_size=4,
                do_sample=True,
                temperature = llm_config_dict[self.task]['temperature'],
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens= llm_config_dict[self.task]['max_new_tokens'],
                truncation=True
            )

            prompt = [
                {"role": "system", "content": pre_prompt},
                {"role": "user", "content": f"Original document: '{data_input}'\nSummary: '{summary}'"}
            ]
            
            llm_prompt_template = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            
            response = llm_pipeline(llm_prompt_template)[0]["generated_text"]
            return response
        except Exception as e:
            print(f"Error in evaluate_summary_openllm: {e}")
            return None

    def extract_scores_from_response(self, response):
        """Extract scores and explanations from the model response."""
        try:
            match = re.search(r"{.*}", response, re.DOTALL)
            if match:
                score_dict_str = match.group(0)
                score_dict = ast.literal_eval(score_dict_str.replace("\n",""))
                return tuple(score_dict.get(metric) for metric in self.metrices) + (score_dict.get('explanation'),)
        except Exception as e:
            print(f"Error in extract_scores_from_response: {e}")
        return tuple([None] * len(self.metrices)) + (None,)

    def evaluate_summaries(self):
        """Main function to evaluate summaries from input files."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        for file in self.input_file_paths:
            print('prompt key', file.split('/')[-1].split('.')[0]+'_'+self.levels)
            # self.pre_prompt = prompt_dict.get(file.split('/')[-1].split('.')[0]+'_'+self.levels)
            prompt_key = file.split('/')[-1].split('.')[0] + '_' + self.levels
            # print('prompt_key: ', prompt_key )
            
            if prompt_dict.get(prompt_key):
                self.pre_prompt = prompt_dict.get(prompt_key)
            else:
                self.pre_prompt = prompt_dict.get(f'general_{self.levels}')

            self.metrices = metrices_dict.get(file.split('/')[-1].split('.')[0])
            print('self.metrices', self.metrices)

            try:
                if file.split('.')[-1] == 'xlsx':
                    data = pd.read_excel(file).iloc[:2]
                elif file.split('.')[-1] == 'csv':
                    data = pd.read_csv(file).iloc[:2]
                print(len(data))
                data.reset_index(drop=True, inplace=True)

                for model_id in self.model_ids:
                # Load model and tokenizer
                    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
                    tokenizer = AutoTokenizer.from_pretrained(model_id)

                    # Use DataParallel for multi-GPU usage  
                    if torch.cuda.device_count() > 1:
                        print(f"Let's use {torch.cuda.device_count()} GPUs!")
                        model = DataParallel(model)

                    model.to(device)                    

                    for summary_column in self.summary_columns:
                        print('data columns :', data.columns)
                        print('summary_column :',summary_column)

                        columns = list(['llm_response'] + [metric for metric in self.metrices] + ['explanation'])

                        if self.is_improved == False:
                            subdir = 'output_llm_evaluation'
                        else:
                            subdir = 'output_improved_llm_evaluation'
                        if self.output_dir == '':
                            outdir = subdir
                        else:
                            outdir = f'{self.output_dir}/{subdir}'
                        os.makedirs(outdir, exist_ok=True)
                        output_path = os.path.join(outdir, os.path.split(file)[1])
                        
                        if self.levels == 'summary-level':
                            for col in columns:
                                if summary_column + '_' + col + f'_{model_id}' not in data.columns:
                                    data[summary_column + '_' + col + f'_{model_id}'] = None
                        elif self.levels == 'summary-sentence-level':
                            for col in columns:
                                if summary_column + '_' + col + f'_{model_id}' + '_sentence_level' not in data.columns:
                                    data[summary_column + '_' + col + f'_{model_id}' + '_sentence_level'] = None
                        
                        for i, row in data.iterrows():
                            if any(pd.isna(row[col]) for col in [col for col in data.columns if not col.startswith(summary_column + '_' +'llm_response' + f'_{model_id}')]):
                            # if any(pd.isna(row[summary_column + '_' + col]) for col in [col for col in columns if not col == 'llm_response']):

                                data_input = row[self.article_column]
                                summary = row[summary_column]
                                is_repeated = any(
                                                    value == summary for column, value in row.items() if 
                                                    column != summary_column)
                                
                                if not is_repeated:
                                    print(f'working on row number {i}')
                                    if self.levels == 'summary-level':
                                        response = self.evaluate_summary_openllm(self.pre_prompt, data_input, summary, model, tokenizer)
                                        if response:
                                            scores = self.extract_scores_from_response(response)
                                            if any(score is None for score in scores):
                                                missing_scores_columns = [[column for column in columns if column != columns[0]][index] for index, score in enumerate(scores) if score is None]
                                                missing_scores_dict = extract_scores_if_missing(response, missing_scores_columns)

                                                scores = [
                                                            missing_scores_dict[[column for column in columns if column != columns[0]][index]] if score is None else score
                                                            for index, score in enumerate(scores)
                                                        ]
                                            print('scores', scores)
                                            for col, score in zip(columns, [response] + list(scores)):
                                                data.at[i,summary_column + '_' + col  + f'_{model_id}'] = score
                                    
                                    elif self.levels == 'summary-sentence-level':
                                        scores_dict = {col: [] for col in columns}

                                        if pd.isna(summary):
                                            summary = ' '
                                        
                                        if not self.summary_sentence_column:
                                            summary_sentences = nltk.sent_tokenize(summary)
                                        else:
                                            summary_sentences = row[self.summary_sentence_column]
                                        # print('summary sentence', summary_sentences)
                                        for sentence in ast.literal_eval(summary_sentences):
                                            response = self.evaluate_summary_openllm(self.pre_prompt, data_input, sentence, model, tokenizer)
                                            if response:
                                                scores = self.extract_scores_from_response(response)
                                                if any(score is None for score in scores):
                                                    missing_scores_columns = [[column for column in columns if column != columns[0]][index] for index, score in enumerate(scores) if score is None]
                                                    missing_scores_dict = extract_scores_if_missing(response, missing_scores_columns)

                                                    scores = [
                                                                missing_scores_dict[[column for column in columns if column != columns[0]][index]] if score is None else score
                                                                for index, score in enumerate(scores)
                                                            ]

                                                for col, score in zip(columns, [response] + list(scores)):
                                                    scores_dict[col].append(score)
                                            
                                        for col, score in scores_dict.items():
                                            data.at[i,summary_column + '_' + col + f'_{model_id}' + '_sentence_level'] = score
                                
                                else:
                                    if self.levels == 'summary-level':
                                        for col in columns:
                                            data.at[i,summary_column + '_' + col + f'_{model_id}'] = data.at[i,summary_column.replace('improved_','') + '_' + col + f'_{model_id}']
                                    
                                    elif self.levels == 'summary-sentence-level':
                                        for col in columns:
                                            data.at[i,summary_column + '_' + col + f'_{model_id}' +  '_sentence_level'] = data.at[i,summary_column.replace('improved_','')+ '_' + col + f'_{model_id}' + '_sentence_level']

                                if output_path.split('.')[-1] == 'xlsx':
                                    data.to_excel(output_path, index=False)
                                elif output_path.split('.')[-1] == 'csv':
                                    data.to_csv(output_path, index=False)
                                
                                print(f"Results saved to {output_path}")
                        
                        data = data.rename(columns={col: summary_column + '_' + col for col in columns})
                        if output_path.split('.')[-1] == 'xlsx':
                            data.to_excel(output_path, index=False)
                        elif output_path.split('.')[-1] == 'csv':
                            data.to_csv(output_path, index=False)

            except Exception as e:
                print(f"Error processing file {file}: {e}")


# Usage
# input_file_paths = ['input_data/arxiv_data.xlsx']
# model_ids = "meta-llama/Meta-Llama-3.1-8B-Instruct" #"meta-llama/Llama-2-7b-chat-hf" # "HuggingFaceH4/zephyr-7b-beta"
# summary_generator = EvaluateSummary(input_file_paths, model_id = model_id)
# summary_generator.evaluate_summaries()
# input_file_paths = '/path/to/your/input/file.csv'
# evaluator = EvaluateSummary(input_file_paths)
# evaluator.evaluate_summaries()