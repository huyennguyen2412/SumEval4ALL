import pandas as pd
import ast
import numpy as np


def aggregate(row,columns):
    for col in columns:
        row[f"{col}_sentence_level"] = np.mean(ast.literal_eval(row[f"{col}_sentence_level"]))
    
    return row

df = pd.read_excel('output_data/arxiv-gov_report_data_split_sentence.xlsx')
columns = ['relevance','factual_consistency']

df_updated = df.apply(lambda row: aggregate(row, columns), axis=1)

df_updated.to_excel('output_data/arxiv-gov_report_data_split_sentence.xlsx')
