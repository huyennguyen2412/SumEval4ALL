import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from add_and_save import update_and_save_df
import nltk

class BleuScore:
    def __init__(self, df, summary_column, article_column, output_file_path=r"output/all_evaluation_scores.csv"):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df
        self.output_file_path = output_file_path
        
    def tokenize(self,text):
        return nltk.word_tokenize(text)

    def calculate_bleu_score(self, tokenized_summary, tokenized_reference):
        # Get embeddings for both summary and reference
        # summary_emb = self.get_embedding(summary)
        # reference_emb = self.get_embedding(reference)
        
        # Calculate cosine similarity
        bleu_score = sentence_bleu(tokenized_reference, tokenized_summary)
        return bleu_score

    def get_score(self):
        nltk.download("punkt_tab")
        nltk.download('wordnet')

        if self.summary_column + '_' + 'auto_eval_bleu' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_bleu'].isna()]
        else:
            rows_to_process = self.df

        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):
            # Compute the BART score
            tokenized_summary = self.tokenize(summary)
            tokenized_reference = self.tokenize(input_text)

            score = self.calculate_bleu_score(tokenized_summary, tokenized_reference)
            print('blue score', row_index, score)
            self.df = update_and_save_df(self.df, rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_bleu', score, self.output_file_path)

        return self.df

