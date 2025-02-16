from summ_eval.summa_qa_utils import QA_Metric, QG_masked
import spacy 
from add_and_save import update_and_save_df
import pandas as pd

class SummaQAScore:
    def __init__(self, df, summary_column='decoded', article_column='article',output_file_path = r"output/all_evaluation_scores.csv", batch_size=8, max_seq_len=384, use_gpu=True, tokenize=True):
        self.summary_column = summary_column
        self.article_column = article_column
        self.df = df
        self.output_file_path = output_file_path

        self.qa_metric = QA_Metric(batch_size=batch_size, max_seq_len=max_seq_len, use_gpu=use_gpu)
        self.question_generator = QG_masked()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.use_gpu = use_gpu
        self.tokenize = tokenize

    def calculate_summa_qa(self, summary, input_text):
        masked_questions, answer_spans = self.question_generator.get_questions(input_text)
        score_dict = self.qa_metric.compute(masked_questions, answer_spans, summary)
        return score_dict

    def get_score(self):
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            print('Downloading the Spacy en_core_web_sm model (this will only happen once)')
            from spacy.cli import download
            download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
        
        if self.summary_column + '_' + 'auto_eval_summa_qa_prob' in self.df.columns:
            rows_to_process = self.df[self.df[self.summary_column + '_' + 'auto_eval_summa_qa_prob'].isna()]
        else:
            rows_to_process = self.df

        for row_index, (summary, input_text) in enumerate(zip(rows_to_process[self.summary_column], rows_to_process[self.article_column])):
            if self.tokenize:
                input_text = nlp(input_text)
            result = self.calculate_summa_qa(summary, input_text)

            self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_summa_qa_prob', result['summaqa_avg_prob'], self.output_file_path)
            self.df = update_and_save_df(self.df , rows_to_process.index[row_index], self.summary_column + '_' + 'auto_eval_summa_qa_fscore', result['summaqa_avg_fscore'], self.output_file_path)
        return self.df
        
    