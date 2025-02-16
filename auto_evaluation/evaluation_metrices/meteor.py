import nltk
from nltk.translate.meteor_score import single_meteor_score
from add_and_save import update_and_save_df
import pandas as pd

class MeteorScore:
    def __init__(self, df, summary_column='decoded', article_column='article', output_file_path = r"output/all_evaluation_scores.csv"):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df
        self.output_file_path = output_file_path

    def tokenize(self,text):
        return nltk.word_tokenize(text)
    
    def get_score(self):
        nltk.download("punkt_tab")
        nltk.download('wordnet')

        if self.summary_column + '_' + 'auto_eval_meteor' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_meteor'].isna()]
        else:
            rows_to_process = self.df
        
        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):

            tokenized_summary = self.tokenize(summary)
            tokenized_reference = self.tokenize(input_text)
            
            score = single_meteor_score(tokenized_reference, tokenized_summary)
            self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_meteor', score, self.output_file_path)

        return self.df