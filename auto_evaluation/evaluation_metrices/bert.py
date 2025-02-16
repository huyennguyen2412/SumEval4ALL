import bert_score
from add_and_save import update_and_save_df
import pandas as pd

class BertScore:
    def __init__(self, df, summary_column='decoded', article_column='article',output_file_path = r"output/all_evaluation_scores.csv",lang='en', model_type='google-bert/bert-base-uncased', num_layers=8, verbose=False, idf=False,
                 nthreads=4, batch_size=64, rescale_with_baseline=False):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df

        self.lang = lang
        self.model_type = model_type
        self.num_layers = num_layers
        self.verbose = verbose
        self.idf = idf
        self.nthreads = nthreads
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline
        self.output_file_path = output_file_path

    def calculate_bert(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        
        all_preds, hash_code = bert_score.score(
            [summary], reference, model_type=self.model_type,
            num_layers=self.num_layers, verbose=self.verbose, idf=self.idf,
            batch_size=self.batch_size, nthreads=self.nthreads,
            lang=self.lang, return_hash=True, rescale_with_baseline=self.rescale_with_baseline
        )
    
        return all_preds
    
    def get_score(self):

        if self.summary_column + '_' + 'auto_eval_bert_precision' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_bert_precision'].isna()]
        else:
            rows_to_process = self.df

        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):

            result = self.calculate_bert(summary, input_text)
            self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_bert_precision', result[0].cpu().item(), self.output_file_path)
            self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_bert_recall', result[1].cpu().item(), self.output_file_path)
            self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_bert_f1', result[2].cpu().item(), self.output_file_path)
        
        return self.df
    