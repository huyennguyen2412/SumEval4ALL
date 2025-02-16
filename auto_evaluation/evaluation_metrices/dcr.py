import pandas as pd
import torch
import textstat
from add_and_save import update_and_save_df

class DcrScore:
    def __init__(self, df, summary_column, article_column, output_file_path=r"output/all_evaluation_scores.csv"):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df
        self.output_file_path = output_file_path

    def get_score(self):
        # bart_scores = []
        if self.summary_column + '_' + 'auto_eval_dcr' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_dcr'].isna()]
        else:
            rows_to_process = self.df

        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):
            # Compute the BART score
            score = textstat.dale_chall_readability_score(summary)
            print('dcr score', row_index, score)
            self.df = update_and_save_df(self.df, rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_dcr', score, self.output_file_path)

        return self.df

