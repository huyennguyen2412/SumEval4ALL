from blanc import BlancHelp, BlancTune
from add_and_save import update_and_save_df
import pandas as pd

class BlancScore:
    def __init__(self, df, summary_column='decoded', article_column='article', output_file_path = r"output/all_evaluation_scores.csv", device='cuda', inference_batch_size=128, finetune_batch_size=24, use_tune=True):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df

        self.device = device
        self.inference_batch_size = inference_batch_size
        self.finetune_batch_size = finetune_batch_size
        self.use_tune = use_tune
        self.output_file_path = output_file_path
        if self.use_tune:
            self.blanc_mod = BlancTune(device=self.device)
        else:
            self.blanc_mod = BlancHelp(device=self.device)

    def calculate_blanc(self, summary, input_text):
        score = self.blanc_mod.eval_once(input_text, summary)
        return score

    def get_score(self):
        if self.summary_column + '_' + 'auto_eval_blanc' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_blanc'].isna()]
        else:
            rows_to_process = self.df

        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):
            result = self.calculate_blanc(summary, input_text)
            self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_blanc',result, self.output_file_path)
        return self.df 