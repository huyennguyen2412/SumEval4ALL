import pandas as pd
from questeval.questeval_metric import QuestEval
from add_and_save import update_and_save_df

class QuestEvalScore:
    def __init__(self, df, summary_column='decoded', article_column='article', output_file_path = r"output/all_evaluation_scores.csv"):
        self.df = df
        self.summary_column = summary_column
        self.article_column = article_column
        self.output_file_path = output_file_path

        # Initialize QuestEval with our custom model
        self.questeval = QuestEval(do_weighter=True, task='summarization')
        
    def calculate_questeval_score(self, source, generated_summary):
        try:
            score = self.questeval.corpus_questeval(sources=[source], hypothesis=[generated_summary])
            return score
        except Exception as e:
            print(f"Error calculating QuestEval score: {e}")
            return None

    def get_score(self):
        if self.summary_column + '_' + 'auto_eval_quest_ex_level' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_quest_ex_level'].isna()]
        else:
            rows_to_process = self.df

        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):
            if input_text and summary:
                score = self.calculate_questeval_score(input_text, summary)
                if score:
                    self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_quest_ex_level', score['ex_level_scores'][0], self.output_file_path)

        return self.df
