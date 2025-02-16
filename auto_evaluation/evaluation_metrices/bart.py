import pandas as pd
import torch
from transformers import BartTokenizer, BartModel
from sklearn.metrics.pairwise import cosine_similarity
from add_and_save import update_and_save_df
from evaluation_metrices.bart_score import BARTScorer

class BartScore:
    def __init__(self, df, summary_column, article_column, output_file_path=r"output/all_evaluation_scores.csv"):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df
        self.output_file_path = output_file_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')

    def calculate_bart_score(self, summary, reference):
        bart_score = self.bart_scorer.score(srcs = [reference], tgts = [summary], batch_size=4)
        return bart_score[0]
    
    def get_score(self):
        # bart_scores = []
        if self.summary_column + '_' + 'auto_eval_bart' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_bart'].isna()]
        else:
            rows_to_process = self.df

        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):
            # Compute the BART score
            score = self.calculate_bart_score(summary, input_text)
            self.df = update_and_save_df(self.df, rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_bart', score, self.output_file_path)

        return self.df

