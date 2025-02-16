import pandas as pd
import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # Transformers' components for loading models and tokenizers
import torch
from torch.amp import autocast
from llm_evaluation.llm_config import llm_config_dict

class GenSummary():
    def __init__(self, input_file_paths, model_id, article_column, 
                 output_dir: str = ''):
        load_dotenv(dotenv_path='config.env')
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.input_file_paths = input_file_paths
        self.output_dir = output_dir
        self.model_id = model_id
        self.article_column = article_column
        self.task = 'generate-summary'
        self.preprompt = f"""You are an expert in summarizing the lengthy document into a short and self-contained summary. 
        The generated summary must NOT be longer than {llm_config_dict[self.task]['max_summ_words']} words. The output only includes the generated summary, nothing else.""" #
        

        
    def generate_summary_openllm(self, data_input, model, tokenizer):

        llm_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task = llm_config_dict[self.task]['task'], #"text-generation", #'text-classification', "text-generation"
            do_sample=True,
            temperature = llm_config_dict[self.task]['temperature'],
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens= llm_config_dict[self.task]['max_new_tokens'],
            truncation = True
            )

        prompt_in_chat_format = [
        {
            "role": "system",
            "content": self.preprompt },
        {
            "role": "user",
            "content": f"Document to summarize: {data_input}"}, ]
        # llm_prompt_template = tokenizer.apply_chat_template(
        #     prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        # )
        # start_time = time.time()
        # summary = llm_pipeline(prompt_in_chat_format, max_length=max_length)[0]["generated_text"]
        summary = llm_pipeline(prompt_in_chat_format)[0]["generated_text"]
        # elapsed_time = time.time() - start_time
        # print(f"Summary generated in {elapsed_time:.2f} seconds.")
        print('completed one row...')
        return summary

    def generate_summaries(self):

        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id,token=self.access_token, model_max_length=llm_config_dict[self.task]['model_max_length'], truncation=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_id,quantization_config=bnb_config,token=self.access_token)
        
        for file in self.input_file_paths:

            if file.split('.')[-1] == 'xlsx':
                data = pd.read_excel(file)
            elif file.split('.')[-1] == 'csv':
                data = pd.read_csv(file)

            data['llm_summary'] = ''

            if self.output_dir == '':
                    outdir = 'output_llm_summary'
            else:
                outdir = f'{self.output_dir}/output_llm_summary'
            os.makedirs(outdir, exist_ok=True)
            output_file_name = os.path.join(outdir, os.path.split(file)[1])
            
            for index, row in data.iterrows(): 
                torch.cuda.empty_cache()  # Clear cache before processing each row
                with autocast('cuda'):
                    # decoded_text = tokenizer.decode(tokenizer.encode(row['article'], max_length=8500, truncation=True), skip_special_tokens=True)
                    # print('input tokens length :',len(tokenizer.encode(decoded_text)))
                    data.loc[index, 'llm_summary'] = self.generate_summary_openllm(
                        data_input=row[self.article_column], model=model, tokenizer=tokenizer)                    
                
                # print(data)
                if output_file_name.split('.')[-1] == 'xlsx':
                    data.to_excel(output_file_name,index=False)
                elif output_file_name.split('.')[-1] == 'csv':
                    data.to_csv(output_file_name,index=False)
               
                print(f"Row {index + 1} processed and saved.")