import sacrebleu
from summ_eval.metric import Metric
import pandas as pd
from add_and_save import update_and_save_df

class CHRFScore(Metric):
    def __init__(self, df, summary_column='decoded', article_column='article',output_file_path = r"output/all_evaluation_scores.csv", ncorder=6, beta=2, n_workers=24, remove_whitespace=True):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df

        self.ncorder = ncorder
        self.beta = beta
        self.n_workers = n_workers
        self.remove_whitespace = remove_whitespace
        self.output_file_path = output_file_path

    def calculate_chrf(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        score = sacrebleu.sentence_chrf(summary, reference, char_order=self.ncorder, word_order=0, \
            beta=self.beta, remove_whitespace=self.remove_whitespace)
        return score.score

    def get_score(self):

        if self.summary_column + '_' + 'auto_eval_chrf' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_chrf'].isna()]
        else:
            rows_to_process = self.df

        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):
            result = self.calculate_chrf(summary, input_text)
            self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_chrf', result, self.output_file_path)
        return self.df