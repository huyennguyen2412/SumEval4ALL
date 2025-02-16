from rouge_score import rouge_scorer
from add_and_save import concat_and_save_df
import pandas as pd
import logging

# Set the logging level to WARNING to suppress INFO messages
logging.getLogger('absl').setLevel(logging.WARNING)

class RougeScore:
    def __init__(self, df, summary_column='decoded', article_column='article', output_file_path = r"output/all_evaluation_scores.csv"):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df
        self.output_file_path = output_file_path
        self.count = 0

    def calculate_rouge(self, summary, references):
        self.count += 1
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        if not isinstance(summary,str):
            summary = ''
        # if isinstance(summary, str):
        return scorer.score(references, summary)
        # return {}

    def get_score(self):

        if self.summary_column + '_' + 'auto_eval_rouge1_precision' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_rouge1_precision'].isna()]
            print('rows to process',len(rows_to_process))
        else:
            rows_to_process = self.df

        # print(handle the missing rows condition in this as well)
        rouge_scores = rows_to_process.apply(lambda row: self.calculate_rouge(row[self.summary_column], row[self.article_column]), axis=1)
        
        # Initialize dictionaries to store total scores
        total_scores = {'rouge1': [[], [], []], 'rouge2': [[], [], []], 'rougeL': [[], [], []]} 
        
        for scores in rouge_scores:
            for metric, values in scores.items():
                # precision, recall, fmeasure = values
                total_scores[metric][0].append(values[0])
                total_scores[metric][1].append(values[1])
                total_scores[metric][2].append(values[2])

        grain_metrics = ['precision', 'recall', 'f1']
        format_results = {}
        for metric, scores in total_scores.items():
            for i, score in enumerate(scores):
                format_results[f'{metric}_{grain_metrics[i]}_{self.summary_column}'] = score

        self.df = concat_and_save_df(self.df,pd.DataFrame(format_results,index = rows_to_process.index),self.output_file_path)
        return self.df
    
    def get_score_rb(self):
        rouge_scores = self.df.apply(lambda row: self.calculate_rouge(row[self.summary_column], row[self.article_column]), axis=1)
        # Initialize dictionaries to store total scores
        total_scores = {'rouge1': [[], [], []], 'rouge2': [[], [], []], 'rougeL': [[], [], []]} 
        for scores in rouge_scores:
            avg_score = {'rouge1': [[], [], []], 'rouge2': [[], [], []], 'rougeL': [[], [], []]}
            for sep_scores in scores:
                for metric, values in sep_scores.items():
                    precision, recall, fmeasure = values
                    avg_score[metric][0].append(values[0])
                    avg_score[metric][1].append(values[1])
                    avg_score[metric][2].append(values[2])
            for metric in total_scores.keys():
                total_scores[metric][0].append(sum(avg_score[metric][0])/11)
                total_scores[metric][1].append(sum(avg_score[metric][1])/11)
                total_scores[metric][2].append(sum(avg_score[metric][2])/11)

        grain_metrics = ['precision_rb', 'recall_rb', 'f1_rb']
        format_results = {}
        for metric, scores in total_scores.items():
            for i, score in enumerate(scores):
                format_results[f'{self.summary_column}_auto_eval_{metric}_{grain_metrics[i]}'] = score
                
        self.df = concat_and_save_df(self.df,format_results,self.output_file_path)
        return format_results