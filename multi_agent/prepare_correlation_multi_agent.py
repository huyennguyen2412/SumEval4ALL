import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, kendalltau, spearmanr
from itertools import chain
from llm_evaluation.llm_evaluation_metrices import metrices_dict
from multi_agent.prepare_for_correlation import PrepareForCorrelation

class CorrelationEvaluator:
    def __init__(self, file_paths, correlation_methods=None, is_saving=True):
        self.file_paths = file_paths
        self.correlation_methods = correlation_methods or ['pearson', 'kendall', 'spearman']
        self.is_saving = is_saving
    
    def calculate_corr(self, x, y, method):
        """Calculates the correlation and p-value based on the chosen method."""
        if method == 'pearson':
            return pearsonr(x, y)
        elif method == 'kendall':
            return kendalltau(x, y)
        elif method == 'spearman':
            return spearmanr(x, y)
        else:
            raise ValueError("Method must be 'pearson', 'kendall', or 'spearman'")
    
    def aggregating_scores(self):
        """Aggregates evaluation scores by 'model'."""
        self.data[self.numeric_columns] = self.data[self.numeric_columns].apply(pd.to_numeric, errors='coerce')
        print('numeric columns', self.numeric_columns)
        if 'model' in self.data.columns:
            df_agg = self.data.groupby('model')[self.numeric_columns].mean().reset_index()
            df_agg[self.numeric_columns] = df_agg[self.numeric_columns].applymap(lambda x: round(x, 4))
            return df_agg[self.numeric_columns]
        else:
            return self.data[self.numeric_columns]

    def run_correlation(self, row_set, col_set, input_file_path, method='pearson', btwn='', is_saving=True):
        """Main method to calculate correlation matrix and p-values."""
        # Initialize empty DataFrames to store correlations and p-values
        corrs = pd.DataFrame(index=row_set, columns=col_set)
        pvals = pd.DataFrame(index=row_set, columns=col_set)

        # Calculate correlations and p-values between row_set and col_set
        for row_col in row_set:
            for col in col_set:
                if self.corr_data[row_col].dtype.kind in 'biufc' and self.corr_data[col].dtype.kind in 'biufc':  # Check for numeric columns
                    corr, pval = self.calculate_corr(self.corr_data[row_col].fillna(0), self.corr_data[col].fillna(0), method)
                    corrs.loc[row_col, col] = round(corr, 3)
                    pvals.loc[row_col, col] = round(pval, 3)
                else:
                    corrs.loc[row_col, col], pvals.loc[row_col, col] = np.nan, np.nan

        # Annotate with significance levels
        p = pvals.applymap(lambda x: ''.join(['*' for t in [0.05, 0.01, 0.001] if x <= t]))

        # Plot heatmap
        plt.figure(figsize=(40, 10))
        sns.heatmap(corrs.astype(float), cmap="RdBu", annot=True, vmin=-1, vmax=1)
        plt.title(f'({method.capitalize()}) Correlation Matrix {btwn}')

        # Optionally save results
        if is_saving:
            output_path = f"output_auto_evaluation_multi_agent_new_models/{os.path.split(input_file_path)[-1].split('.')[0]}/{os.path.split(input_file_path)[-1].split('.')[0]}_{method}_correlation_results-{btwn}"
            corrs_p = corrs.round(3).astype(str) + p
            corrs_p.to_csv(output_path + '.csv')
            plt.savefig(output_path + '.png', format='png')

        return corrs, p

    def run_all_correlations_multi_agent(self):
        """Runs correlation evaluation across all files."""
        for file in self.file_paths:
            print('FILE NAME', file)
            if file.split('.')[-1] == 'xlsx':
                self.data = pd.read_excel(file)
            elif file.split('.')[-1] == 'csv':
                self.data = pd.read_csv(file)
            print('Lenght of actual data',len(self.data))

            # Define models and metrics
            # models = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "Saxo/Linkbricks-Horizon-AI-Avengers-V6-32B", "tanliboy/lambda-qwen2.5-14b-dpo-test"]
            models = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "Saxo/Linkbricks-Horizon-AI-Avengers-V6-32B","microsoft/phi-4","Qwen/Qwen2-7B-Instruct"]
            metrices = metrices_dict.get(file.split('/')[-1].split('.')[0])

            summary_columns = [col for col in self.data.columns if col.endswith('summary') and not col.startswith('llm_')]

            if len(summary_columns)>1:
                self.corr_data = PrepareForCorrelation(file,'',models).prepare_data(r'output_multi_agent/output_llm_evaluation/merged_patent_data.xlsx')
                print('patent_data_columns',self.corr_data.columns)
                rows = [col for col in self.corr_data.columns if not 'human' in col]  
                print('rows',rows)
                columns = [col for col in self.corr_data.columns if 'human' in col]
                print('columns',columns)
                self.corr_data[columns + rows] = self.corr_data[columns + rows].apply(pd.to_numeric, errors='coerce')

                for method in self.correlation_methods:
                    self.run_correlation(rows, columns, method=method, input_file_path = file, btwn=f'human_vs_models', is_saving=True)

            # Determine evaluators
            else:
                evaluators = (
                    ['human'] if any("human_" in col for col in self.data.columns if col.endswith('_score')) 
                    else ['avg_expert', 'avg_turker'] if any("avg_" in col for col in self.data.columns if col.endswith('_score')) 
                    else [col.replace('_summary', '') for col in self.data.columns if col.endswith('_summary') and 'llm_' not in col] 
                )  

                evaluator_score_columns = {evaluator: [f"{evaluator}_{metric}_score" for metric in metrices if f"{evaluator}_{metric}_score" in self.data.columns] for evaluator in evaluators}
                model_score_columns = {model: [f"{summary_column}_{metric}_{model}" for metric in metrices for summary_column in summary_columns if f"{summary_column}_{metric}_{model}" in self.data.columns] for model in models}

                rows = list(chain.from_iterable(model_score_columns.values()))
                rows.extend([f'{agg}_{metric}' for metric in metrices for agg in ['average', 'mode']])
                print('rows',rows) 
                columns = list(evaluator_score_columns.keys())

                self.numeric_columns = [item for evaluator in columns for item in evaluator_score_columns[evaluator]] + rows

                # Aggregating evaluation scores
                self.corr_data = self.aggregating_scores()
                print('Lenght of aggregated data', len(self.corr_data))

                # Calculate correlations
                for evaluator in columns:
                    for method in self.correlation_methods:
                        self.run_correlation(rows, evaluator_score_columns[evaluator], file, method=method, btwn=f'{evaluator}_vs_models', is_saving=self.is_saving)

# Example usage:
file_paths = [
    # r'output_auto_evaluation_multi_agent_new_models/arxiv_data/arxiv_data.xlsx',
    # r'output_auto_evaluation_multi_agent_new_models/gov_report_data/gov_report_data.xlsx',
    r'output_auto_evaluation_multi_agent_new_models/patent_sum_eval_data/patent_sum_eval_data.xlsx',
    # r'output_auto_evaluation_multi_agent_new_models/qags_cnn_dm_data/qags_cnn_dm_data.csv',
    # r'output_auto_evaluation_multi_agent_new_models/qags_x_sum_data/qags_x_sum_data.csv',
    # r'output_auto_evaluation_multi_agent_new_models/summ_eval_data/summ_eval_data.csv',
    # r'output_multi_agent_new_models/output_llm_evaluation/tldr_data.xlsx'
]

evaluator = CorrelationEvaluator(file_paths)
evaluator.run_all_correlations_multi_agent()


##### Usage python3 -m multi_agent.prepare_correlation_multi_agent.py