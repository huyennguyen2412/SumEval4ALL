from summac.model_summac import SummaCZS, SummaCConv
from add_and_save import concat_and_save_df
import pandas as pd

class SummaCScore:
        def __init__(self, df, summary_column='decoded', article_column='article', output_file_path = r"output/all_evaluation_scores.csv", device = 'cuda'):
                self.summary_column = summary_column
                self.article_column = article_column
                self.df = df
                self.device = device
                self.output_file_path = output_file_path
        
        def get_score(self):
                if self.summary_column + '_' + 'auto_eval_summac_zs' in self.df.columns:
                        rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_summac_zs' ].isna()]
                else:
                        rows_to_process = self.df

                model_zs = SummaCZS(granularity="sentence", model_name="vitc", device=self.device)
                model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=self.device, start_file="default", agg="mean")
                
                score_zs = model_zs.score(rows_to_process[self.article_column].to_list(), rows_to_process[self.summary_column].to_list())
                score_conv = model_conv.score(rows_to_process[self.article_column].to_list(), rows_to_process[self.summary_column].to_list())
                
                self.df = concat_and_save_df(self.df,pd.DataFrame({f'{self.summary_column}_auto_eval_summac_zs': score_zs['scores'], f'{self.summary_column}_auto_eval_summac_conv': score_conv['scores']}, index = rows_to_process.index),self.output_file_path)
                return self.df