import os
import pandas as pd
import requests
import zipfile
from summ_eval.s3_utils import rouge_n_we
from summ_eval.metric import Metric
from add_and_save import update_and_save_df

class RougeWeScore(Metric):
    def __init__(self, df, summary_column='decoded', article_column='article',output_file_path = r"output/all_evaluation_scores.csv", n_gram=3, n_workers=24, tokenize=True):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df

        self.glove_embeddings_path = './glove.6B/glove.6B.50d.txt'
        self.download_glove_embeddings()
        self.word_embeddings = self.load_glove_embeddings(self.glove_embeddings_path)
        self.n_gram = n_gram
        self.n_workers = n_workers
        self.tokenize = tokenize
        self.output_file_path = output_file_path

    def load_glove_embeddings(self, file_path):
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = list(map(float, parts[1:]))
                embeddings[word] = vector
        return embeddings
    
    def download_glove_embeddings(self):
        # URL to the GloVe 6B file
        url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        output_path = './glove.6B.zip'
        
        # Download GloVe embeddings if not already downloaded
        if not os.path.exists(output_path):
            print("Downloading GloVe embeddings...")
            response = requests.get(url)
            with open(output_path, 'wb') as file:
                file.write(response.content)
            print("Download completed.")

        # Extract the ZIP file
        if not os.path.exists('./glove.6B'):
            print("Extracting GloVe embeddings...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall('./glove.6B')
            print("Extraction completed.")


    def calculate_rouge_we(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        if not isinstance(summary, list):
            summary = [summary]
        score = rouge_n_we(summary, reference, self.word_embeddings, self.n_gram, 
                        return_all=True, tokenize=self.tokenize)
        return score
        
    def get_score(self):
        if self.summary_column + '_' + 'auto_eval_rouge_we_p' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_rouge_we_p'].isna()]
        else:
            rows_to_process = self.df
        
        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):
            result = self.calculate_rouge_we(summary, input_text)

            self.df = update_and_save_df(self.df ,rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_rouge_we_p', result[0], self.output_file_path)
            self.df = update_and_save_df(self.df ,rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_rouge_we_r', result[1], self.output_file_path)
            self.df = update_and_save_df(self.df ,rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_rouge_we_f', result[2], self.output_file_path)
    
        return self.df
    


