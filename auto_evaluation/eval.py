import pandas as pd
import ast
import os
from evaluation_metrices.rouge import RougeScore
from evaluation_metrices.bert import BertScore
from evaluation_metrices.summa_c import SummaCScore
from evaluation_metrices.summa_qa import SummaQAScore
from evaluation_metrices.chrf import CHRFScore
from evaluation_metrices.meteor import MeteorScore
from evaluation_metrices.blanc import BlancScore
from evaluation_metrices.rouge_we import RougeWeScore
from evaluation_metrices.bart import BartScore
from evaluation_metrices.qa_eval import QAEvalScore
from evaluation_metrices.quest_eval import QuestEvalScore
from evaluation_metrices.bleu import BleuScore
from evaluation_metrices.fre import FreScore
from evaluation_metrices.dcr import DcrScore

class AutoEvaluation():
  def __init__(self, input_file_paths, article_column, summary_columns):
    self.input_file_paths = input_file_paths
    self.summary_columns = summary_columns
    self.article_column = article_column
    self.class_dict = {
                'rouge_score':RougeScore,
                'bert_score':BertScore,
                'summa_c_scores':SummaCScore,
                'summa_qa_scores':SummaQAScore,
                'chrf_score':CHRFScore,
                'meteor_score':MeteorScore,
                'blanc_score':BlancScore,
                'rouge_we_score':RougeWeScore,
                'bart_score':BartScore,
                'qa_eval':QAEvalScore,
                'quest_eval':QuestEvalScore,
                'bleu_score':BleuScore,
                'fre_score':FreScore,
                'dcr_score':DcrScore
                }

  def run_auto_evaluation(self, evaluations_to_perform):
    for input_file_path in self.input_file_paths:
      output_file_path = os.path.join('output_auto_evaluation',input_file_path.split('/')[-1])
      if input_file_path.split('.')[-1] == 'xlsx':
        self.input_data = pd.read_excel(input_file_path)
      elif input_file_path.split('.')[-1] == 'csv':
        self.input_data = pd.read_csv(input_file_path)

      for summary_column in self.summary_columns:
        print('summary_column :',summary_column)
        self.input_data[summary_column] =  self.input_data[summary_column].apply(lambda x: '' if pd.isna(x) else x)
        # all_scores = {}
        for matrix in evaluations_to_perform:
            print('matrix :',matrix)
            self.input_data = self.class_dict[matrix](self.input_data, summary_column = summary_column, article_column = self.article_column, output_file_path= output_file_path).get_score()
            print('columns :',self.input_data.columns)