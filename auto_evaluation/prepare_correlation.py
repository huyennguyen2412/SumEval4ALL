import subprocess
import sys
import pandas as pd
import ast
import numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import os

class RunCorrelation:
    def __init__(self, input_file_paths, evaluators, model_column, correlation_methods, 
                metrics_to_compare, human_metrices, agg_human_metrices, group_by_model):
        # Initialize with the file paths and correlation methods
        # self.run_requirements()
        self.input_file_paths = input_file_paths
        self.correlation_methods = correlation_methods
        self.evaluators = evaluators
        self.model_column = model_column
        self.metrics_to_compare = metrics_to_compare
        self.human_metrices = human_metrices
        self.agg_human_metrices = agg_human_metrices
        self.group_by_model = group_by_model

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

    def run_correlation(self, row_set, col_set, input_file_path,method='pearson', btwn='',is_saving=True):
        """Main method to calculate correlation matrix and p-values."""
        # Initialize empty DataFrames to store correlations and p-values
        corrs = pd.DataFrame(index=row_set, columns=col_set)
        pvals = pd.DataFrame(index=row_set, columns=col_set)

        # Calculate correlations and p-values between row_set and col_set
        for row_col in row_set:
            for col in col_set:
                if self.corr_data[row_col].dtype.kind in 'biufc' and self.corr_data[col].dtype.kind in 'biufc':  # Check for numeric columns
                    corr, pval = self.calculate_corr(self.corr_data[row_col].fillna(0), self.corr_data[col].fillna(0), method)
                    corrs.loc[row_col, col] = round(corr,3)
                    pvals.loc[row_col, col] = round(pval,3)
                else:
                    corrs.loc[row_col, col], pvals.loc[row_col, col] = np.nan, np.nan

        # Annotate with significance levels
        p = pvals.applymap(lambda x: ''.join(['*' for t in [0.05, 0.01, 0.001] if x <= t]))

        # Plot heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(corrs.astype(float), cmap="RdBu", annot=True, vmin=-1, vmax=1)
        plt.title(f'({method.capitalize()}) Correlation Matrix {btwn}')
        plt.show()

        # Optionally save results
        if is_saving:
            output_path = f"output_auto_evaluation/{self.evaluator}/{os.path.split(input_file_path)[-1]}_{method}_correlation_results-{btwn}"
            corrs_p = corrs.round(3).astype(str) + p
            corrs_p.to_csv(output_path + '.csv')
            plt.savefig(output_path + '.png', format='png')

        return corrs, p

    def run_all_correlations(self):

        for input_file_path in self.input_file_paths:
            
            if input_file_path.split('.')[-1] == 'xlsx':
                self.llm_scores = pd.read_excel(input_file_path)
            elif input_file_path.split('.')[-1] == 'csv':
                self.llm_scores = pd.read_csv(input_file_path)

            # if self.agg_human_metrices == True:
            #     for metric in self.human_metrices:
            #         self.llm_scores[f'human_{metric}'] = self.llm_scores[[col for col in self.llm_scores.columns if any(col.endswith(suffix.split('_')[0]+'_score') for suffix in [metric])]].mean(axis=1)
            #         # self.llm_scores['human_factual_score'] = self.llm_scores[[col for col in self.llm_scores.columns if any(col.endswith(suffix) for suffix in ['factual_score'])]].mean(axis=1)

            # self.llm_scores['human_score'] = self.llm_scores['human_score'].apply(ast.literal_eval)
            # self.llm_scores[['human_overall', 'human_accuracy', 'human_coverage', 'human_coherence','compatible']] = pd.json_normalize(self.llm_scores['human_score'])

            # self.llm_scores['avg_expert_annotations'] = self.llm_scores['avg_expert_annotations'].apply(ast.literal_eval)
            # self.llm_scores[['avg_expert_coherence', 'avg_expert_consistency', 'avg_expert_fluency', 'avg_expert_relevance']] = pd.json_normalize(self.llm_scores['avg_expert_annotations'])
            # self.llm_scores['avg_turker_annotations'] = self.llm_scores['avg_turker_annotations'].apply(ast.literal_eval)
            # self.llm_scores[['avg_turker_coherence', 'avg_turker_consistency', 'avg_turker_fluency', 'avg_turker_relevance']] = pd.json_normalize(self.llm_scores['avg_turker_annotations'])
            
            
            numeric_columns = self.llm_scores.select_dtypes(include=[np.number]).columns
            print(numeric_columns)
            if self.group_by_model == True:    
                self.corr_data = self.llm_scores.groupby(self.model_column)[numeric_columns].mean().reset_index()
            else:
                self.corr_data = self.llm_scores[numeric_columns]

            metrices_dict = {}
            for self.evaluator in self.evaluators:

                """Runs correlations for all methods."""
                human_metrices = [f'{self.evaluator}_{metric}_score' for metric in self.human_metrices]
                print('human_metrices', human_metrices)
                # llm_eval_metrices = self.llm_scores.columns[self.llm_scores.columns.get_loc('llm_raw_response')+1: self.llm_scores.columns.get_loc('references_1')]
                llm_eval_metrices = [f'llm_summary_{metric}' for metric in self.human_metrices]
                # llm_eval_metrices = [col for col in self.llm_scores.columns if 'llm_' in col]
                print('llm_eval_metrices', llm_eval_metrices)
                
                metrices_dict['human'] = human_metrices
                metrices_dict['llm_eval'] = llm_eval_metrices
                
                if 'llm_response_sentence_level' in self.llm_scores.columns:
                    # llm_sentence_eval_metrices = self.llm_scores.columns[self.llm_scores.columns.get_loc('llm_response_sentence_level')+1: self.llm_scores.columns.get_loc('explanation_sentence_level')]
                    llm_sentence_eval_metrices = [f'llm_summary_{metric}_sentence_level' for metric in self.human_metrices]
                    print('llm_sentence_eval_metrices',llm_sentence_eval_metrices)
                    # auto_eval_metrices = self.llm_scores.columns[self.llm_scores.columns.get_loc('explanation_sentence_level')+1:]
                    # print('auto_eval_metrices',auto_eval_metrices)
                    
                    metrices_dict['llm_sentence_eval'] = llm_sentence_eval_metrices

                
                    # auto_eval_metrices = self.llm_scores.columns[self.llm_scores.columns.get_loc(llm_eval_metrices[-1])+2:self.llm_scores.columns.get_loc('avg_expert_coherence')]
                # auto_eval_metrices = self.llm_scores.columns[self.llm_scores.columns.get_loc('rouge1_precision'):self.llm_scores.columns.get_loc('avg_expert_coherence')]
                auto_eval_metrices = [col for col in self.corr_data.columns if col.contains(f'summary_auto_eval_')]
                
                print('auto_eval_metrices',auto_eval_metrices)

                metrices_dict['auto_eval'] = auto_eval_metrices 

                for metrices in self.metrics_to_compare:
                    print(metrices)
                    if metrices == 'all':
                        row_set = []
                        col_set = human_metrices
                        for metric in metrices_dict.keys():
                            if metric != 'human':
                                row_set.extend(metrices_dict[metric])
                        # row_set = [list(metrices_dict[key]) for key in metrices_dict.keys() if key != 'human']
                        # row_set = list(llm_eval_metrices) + list(llm_sentence_eval_metrices) + list(auto_eval_metrices)
                        print('all', col_set, row_set)
                    else:
                        col_set = []
                        row_set = []
                        for metric in metrices.split('_vs_')[0].split('-'):
                            col_set.extend(metrices_dict[metric])  # Append values to col_variables
                            print('metric', col_set)

                        for metric in metrices.split('_vs_')[1].split('-'):   
                            row_set.extend(metrices_dict[metric])  # Append values to row_variables
                            print('metric', row_set)
                    # Define column set and row set
                    # col_set = self.llm_scores.columns
                    # col_set = [col for col in self.llm_scores.columns if any(col.endswith(suffix) for suffix in ['relevance_score','factual_score'])]
                    # print(col_set)
                    # col_set = ['human_overall', 'human_accuracy', 'human_coverage', 'human_coherence']
                    # row_set= ['llm_coherence','llm_consistency','llm_relevance','llm_fluency']
                    # row_set = [item for item in self.llm_scores.columns if item.startswith('llm_')][1:] 
                    # row_set = self.llm_scores.columns[self.llm_scores.columns.get_loc('llm_response')+1: self.llm_scores.columns.get_loc('explanation_sentence_level')]
                    # row_set = self.llm_scores.columns[self.llm_scores.columns.get_loc('llm_response')+1: self.llm_scores.columns.get_loc('human_relevance_score')]
                    # row_set = [item for item in row_set if item not in ['explanation','llm_response_sentence_level','explanation_sentence_level']] s
                    # print(row_set)

                    # Run the correlation for each method
                    for method in self.correlation_methods:
                        self.run_correlation(row_set, col_set, method=method, input_file_path = input_file_path, btwn=metrices, is_saving=True)

                # col_set = ['avg_expert_coherence', 'avg_expert_consistency', 'avg_expert_fluency', 'avg_expert_relevance']
                
                # row_set = self.llm_scores.columns[self.llm_scores.columns.get_loc('llm_coherence'): self.llm_scores.columns.get_loc('avg_expert_coherence')]
                # # row_set = ['llm_coherence','llm_consistency','llm_relevance','llm_fluency']
                # row_set = [item for item in self.llm_scores.columns if item.startswith('llm_')]
                # print(row_set)
                # # row_set = [item for item in row_set if not item.startswith('references_')]
                # # Run the correlation for each method
                # for method in self.correlation_methods:
                #     self.run_correlation(row_set, col_set, method=method, btwn='avg_expert_vs_auto_eval-llm_eval', is_saving=True)
