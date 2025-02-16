from add_and_save import concat_and_save_df
import nltk
import pandas as pd
from qaeval import QAEval
nltk.download('averaged_perceptron_tagger_eng')

class QAEvalScore:
    def __init__(self, df, summary_column='decoded', article_column='article', output_file_path = r"output/all_evaluation_scores.csv", model_name="all-MiniLM-L6-v2"):
        
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df
        self.output_file_path = output_file_path

    def get_score(self):
        
        if self.summary_column + '_' + 'auto_eval_qa_eval_f1' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_qa_eval_f1'].isna()]
        else:
            rows_to_process = self.df

        # Iterate over each summary-reference pair and compute the score
        qa_evaluator = QAEval(generation_model_path = r'/raid/huyen/SumDS/cnn_dm/new_scripts/question_generation_model', answering_model_dir = r'/raid/huyen/SumDS/cnn_dm/new_scripts/question_answering_model')

        scores = qa_evaluator.score_batch(summaries=rows_to_process[self.summary_column], references_list=rows_to_process[self.article_column])
        flattened_scores = [item['qa-eval'] for item in scores]

        result_df = pd.DataFrame(flattened_scores,index = rows_to_process.index)
        result_df.columns = [self.summary_column + '_' + 'auto_eval_qa_eval_' + col for col in result_df.columns]
        self.df = concat_and_save_df(self.df,result_df,self.output_file_path)
        return self.df
