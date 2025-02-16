import os
import numpy as np
import pandas as pd
import ast
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

if not os.path.exists('input'):
    os.makedirs('input')

for file in os.listdir('raw_input'):
    
    if file == 'arxiv_data.xlsx' or file == 'gov_report_data.xlsx':
        data = pd.read_excel(os.path.join('raw_input',file))
        human_metrices = ['relevance_score','factual_score']
        for metric in human_metrices:
            data[f'human_{metric}'] = data[[col for col in data.columns if any(col.endswith(suffix.split('_')[0]+'_score') for suffix in [metric])]].mean(axis=1)
    
        # data['summary_sentences'] = data['summary'].apply(sent_tokenize)
        data['summary_sentences'] = data['summary'].apply(lambda x: sent_tokenize(x) if isinstance(x, str) else None)
        columns_order = ['model','article_content', 'summary_sentences', 'summary', 'human_relevance_score', 'human_factual_score']
        
        data = data[columns_order]
        data.to_excel(os.path.join('input','eval_datasets',file),index=False)
    
    if file == 'patent_sum_eval_data.xlsx':
        data = pd.read_excel(os.path.join('raw_input',file))
        data['article_content'] = data['abstract'] + data['claims']
        model_mapping = {
                            'summary_1': 'hupd-t5-base',
                            'summary_2': 'xl_net',
                            'summary_3': 'bart',
                            'summary_4': 'long-t5-tglobal-base-16384-book-summary-2',
                            'summary_5': 'gpt35'
                        }
        columns_rename_dict = {
            col: model_mapping['_'.join(col.split('_')[1:])] + '_' + col.split('_')[0]
            for col in data.columns if 'summary' in col.split('_')
        }

        columns_rename_dict.update({
            col: col+'_summary'
            for col in data.columns if col in model_mapping.values()
        })

        data = data.rename(columns = columns_rename_dict)
        
        evaluators = ['bart','gpt35','hupd-t5-base','long-t5-tglobal-base-16384-book-summary-2','xl_net']
        human_metrices = ['accuracy','overal','coverage','clarity']

        data.columns = [
                            col + '_score' if any(metric in col for metric in ['_accuracy', '_overall', '_coverage', '_clarity']) else col
                            for col in data.columns
                        ]

        columns_order = ['article_content', 'bart_summary', 'gpt35_summary', 'hupd-t5-base_summary', 
                        'long-t5-tglobal-base-16384-book-summary-2_summary', 'xl_net_summary','hupd-t5-base_accuracy_score', 
                        'xl_net_accuracy_score','bart_accuracy_score','long-t5-tglobal-base-16384-book-summary-2_accuracy_score',
                        'gpt35_accuracy_score', 'hupd-t5-base_overall_score','xl_net_overall_score', 'bart_overall_score',
                        'long-t5-tglobal-base-16384-book-summary-2_overall_score','gpt35_overall_score', 
                        'hupd-t5-base_coverage_score', 'xl_net_coverage_score', 'bart_coverage_score',
                        'long-t5-tglobal-base-16384-book-summary-2_coverage_score', 'gpt35_coverage_score', 
                        'hupd-t5-base_clarity_score','xl_net_clarity_score', 'bart_clarity_score',
                        'long-t5-tglobal-base-16384-book-summary-2_clarity_score','gpt35_clarity_score']
        
        data = data[columns_order]
        data.to_excel(os.path.join('input','eval_datasets',file),index=False)

    if file == 'qags_cnn_dm_data.csv' or file == 'qags_x_sum_data.csv':

        def average_consecutive(lst, window_size=3):
            # Split the list into chunks of size 3 and compute the average of each chunk
            return [sum(lst[i:i + window_size]) / window_size for i in range(0, len(lst), window_size)]
        
        data = pd.read_csv(os.path.join('raw_input',file))
        data = data.rename(columns = {'article':'article_content','concatenated_summary':'summary'})
        data['human_factual_consistency_score_sentence_level'] = data['responses'].apply(ast.literal_eval).apply(lambda x: [1 if i == 'yes' else 0 for i in x]).apply(lambda x: average_consecutive(x))
       
        data['human_factual_consistency_score'] = data['human_factual_consistency_score_sentence_level'].apply(lambda x: np.mean(x))

        columns_order = ['article_content','summary_sentences','summary','human_factual_consistency_score_sentence_level','human_factual_consistency_score']

        data = data[columns_order]
        data.to_csv(os.path.join('input','eval_datasets',file),index=False)

    if file == 'tldr_data.csv':
        data = pd.read_csv(os.path.join('raw_input',file)).reset_index(drop = True)
        data = data.rename(columns = {'article':'article_content','abstract':'summary'})

        data['human_score'] = data['human_score'].apply(ast.literal_eval)
        data[['human_overall_score', 'human_accuracy_score', 'human_coverage_score', 'human_coherence_score','human_compatible_score']] = pd.json_normalize(data['human_score'])

        columns_order = ['article_content','summary','human_overall_score', 'human_accuracy_score', 'human_coverage_score', 'human_coherence_score']
        
        data = data[columns_order]
        data.to_excel(os.path.join('input','eval_datasets','tldr_data.xlsx'),index=False)
    
    if file == 'summ_eval_data.csv':
        data = pd.read_csv(os.path.join('raw_input',file))
        data = data.rename(columns = {'model_id':'model','article':'article_content','decoded':'summary',})

        data['avg_expert_annotations'] = data['avg_expert_annotations'].apply(ast.literal_eval)
        data[['avg_expert_coherence_score', 'avg_expert_consistency_score', 'avg_expert_fluency_score', 'avg_expert_relevance_score']] = pd.json_normalize(data['avg_expert_annotations'])
        data['avg_turker_annotations'] = data['avg_turker_annotations'].apply(ast.literal_eval)
        data[['avg_turker_coherence_score', 'avg_turker_consistency_score', 'avg_turker_fluency_score', 'avg_turker_relevance_score']] = pd.json_normalize(data['avg_turker_annotations'])
            
        columns_order = ['model','article_content','summary','avg_expert_coherence_score', 'avg_expert_consistency_score',
                        'avg_expert_fluency_score', 'avg_expert_relevance_score', 'avg_turker_coherence_score', 
                        'avg_turker_consistency_score', 'avg_turker_fluency_score', 'avg_turker_relevance_score']
        
        data = data[columns_order]
        data.to_csv(os.path.join('input','eval_datasets',file),index=False)