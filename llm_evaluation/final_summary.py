### Use this code to improve and score on the summaries
import os
import pandas as pd
import transformers
from transformers import pipeline
# import my_tokens
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
from llm_evaluation.llm_config import llm_config_dict
from llm_evaluation.llm_evaluation_metrices import metrices_dict
# import utils



class ImproveSummary():
    def __init__(self, input_file_paths, model_id, article_column, summary_column, explanation_column, metric_to_compare):
        # os.environ['HF_TOKEN'] = my_tokens.my_hf_token
        load_dotenv()
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.input_file_paths = input_file_paths
        self.model_id = model_id
        self.article_column = article_column
        self.summary_column = summary_column
        self.explanation_column = explanation_column
        self.metric_to_compare = metric_to_compare

        self.preprompt = f"""I would like you to improve the provided summary based on the evaluation feedback given to that summary. 
            Refer to the original document to make improvements. The output must be only the improved summary, nothing else.""" #
        self.task = 'improve-generate-summary'


    def improve_summary_openllm(self,data_input, summary2improve, eval_feedback, model, tokenizer):
        
        llm_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task=llm_config_dict[self.task]['task'],#"text-generation", #'text-classification', "text-generation"
            do_sample=True,
            temperature= llm_config_dict[self.task]['temperature'],
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=llm_config_dict[self.task]['max_new_tokens'],
            truncation = True
        )
    
        prompt_in_chat_format = [
        {
            "role": "system",
            "content": self.preprompt },
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
        
        return summary


    def improve_summaries(self):
        """"
        model_name: llama3, gpt3.5
        """
        # in_file_path = r'/home/lavanya/lavanya/Sum_Eval/new_dataset_outputs/optimalPrompt_CNN_updated_eval_scores.csv' 
        # with open (in_file_path, 'r') as f:
        #     data = pd.read_csv(f)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(self.model_id,token=self.access_token, quantization_config=bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id,token=self.access_token, model_max_length=llm_config_dict[self.task]['model_max_length'],truncation = True)


        for file in self.input_file_paths:

            out_file_path = os.path.join('output_improved_llm_summary',os.path.split(file)[1])

            if file.split('.')[-1] == 'xlsx':
                data = pd.read_excel(file)
            elif file.split('.')[-1] == 'csv':
                data = pd.read_csv(file)
            # data = pd.read_excel(file).iloc[:5]
            # data = data[:10]
                
            print(data.columns)

            # improved_summaries = []
            for i, row in data.iterrows():
                if row[f'{self.summary_column}_{self.metric_to_compare}_{self.model_id}'] < 4:
                # if row[f'{self.metric_to_compare}'] < 4:
                    improved_summary = self.improve_summary_openllm(data_input=row[self.article_column],
                                                                            summary2improve=row[self.summary_column],
                                                                            eval_feedback= row[f'{self.explanation_column}_{self.model_id}'],
                                                                            model = model,
                                                                            tokenizer=tokenizer
                                                                            )
                    # improved_summaries.append(improved_summary)
                    data.loc[i, f'improved_llm_summary'] = improved_summary
                else:
                    data.loc[i, f'improved_llm_summary'] = row[self.summary_column]

                
                if out_file_path.split('.')[-1] == 'xlsx':
                    data.to_excel(out_file_path, index=False)
                elif out_file_path.split('.')[-1] == 'csv':
                    data.to_csv(out_file_path, index=False)

                print(f'Completed row: {i}')
            print('done for one file')