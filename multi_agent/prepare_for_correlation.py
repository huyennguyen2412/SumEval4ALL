# import pandas as pd
from llm_evaluation.llm_evaluation_metrices import metrices_dict

# auto_eval_data = pd.read_excel(r'output_auto_evaluation/patent_sum_eval_data_new.xlsx')

# llm_eval_data = pd.read_excel(r'output_llm_evaluation/patent_sum_eval_data_summary-level.xlsx')

# models = ['bart_summary', 'gpt35_summary', 'hupd-t5-base_summary', 'long-t5-tglobal-base-16384-book-summary-2_summary', 'xl_net_summary']

# evaluation_metrices = metrices_dict.get(file.split('/')[-1].split('.')[0])

# auto_eval_cols = []
# for model in models:
#     for col in auto_eval_data.columns:
#         if col.endswith(f'_{model}'): 
#             auto_eval_cols.append(col)

# llm_eval_cols = []
# for model in models:
#     for col in llm_eval_data.columns:
#         if col.endswith(f'_{model}'): 
#             llm_eval_cols.append(col)

# human_eval_cols = []
# for model in models:
#     for col in llm_eval_data.columns:
#         if col.startswith(f"{model[:-len('_summary')]}_") and '_summary' not in col: 
#             human_eval_cols.append(col)

# print('human_eval_cols',human_eval_cols)
# print('llm_eval_cols',llm_eval_cols)
# print('auto_eval_cols',auto_eval_cols)

# final_cols_data = pd.concat([llm_eval_data[human_eval_cols + llm_eval_cols], auto_eval_data[auto_eval_cols]], axis=1)

# print(final_cols_data.columns)

# final_cols_data = final_cols_data.T

# filtered_data = final_cols_data[~final_cols_data.index.str.contains('llm_response|explanation', case=False, na=False)]

# mean_values = filtered_data.mean(axis=1)

# human_columns = ['human_'+metric for metric in evaluation_metrices]

# auto_eval_columns = []
# [auto_eval_columns.append(col.replace(f"_{model}", '')) 
#  for col in auto_eval_cols
#  for model in models
#  if f"_{model}" in col and col.replace(f"_{model}", '') not in auto_eval_columns]

# llm_eval_columns = ['llm_'+metric for metric in evaluation_metrices]

# grouped_data_columns = human_columns + llm_eval_columns + auto_eval_columns

# human_df = pd.DataFrame(index=[model[:-len('_summary')] for model in models], columns=human_columns)

# def fill_human_df(human_df, mean_values):
#     for index in human_df.index:
#         # Extract model's prefix (e.g., 'bart' from 'bart_accuracy')
#         # Set values for each column based on the model's corresponding values in mean_values
#         human_df.loc[index, 'human_accuracy'] = mean_values.get(f'{index}_accuracy', None)
#         human_df.loc[index, 'human_overall'] = mean_values.get(f'{index}_overall', None)
#         human_df.loc[index, 'human_coverage'] = mean_values.get(f'{index}_coverage', None)
#         human_df.loc[index, 'human_clarity'] = mean_values.get(f'{index}_clarity', None)
    
#     return human_df

# # Fill the human_df based on mean_values
# human_df = fill_human_df(human_df, mean_values)


# llm_eval_df = pd.DataFrame(index=[model[:-len('_summary')] for model in models], columns= llm_eval_columns)


# def fill_llm_eval_df(llm_eval_df, mean_values):
#     for index in llm_eval_df.index:
#         # Extract model's prefix (e.g., 'bart' from 'bart_accuracy')
#         # Set values for each column based on the model's corresponding values in mean_values
#         llm_eval_df.loc[index, 'llm_accuracy'] = mean_values.get(f'accuracy_{index}_summary', None)
#         llm_eval_df.loc[index, 'llm_overall'] = mean_values.get(f'overall_{index}_summary', None)
#         llm_eval_df.loc[index, 'llm_coverage'] = mean_values.get(f'coverage_{index}_summary', None)
#         llm_eval_df.loc[index, 'llm_clarity'] = mean_values.get(f'clarity_{index}_summary', None)
    
#     return llm_eval_df

# # Fill the human_df based on mean_values
# llm_eval_df = fill_llm_eval_df(llm_eval_df, mean_values)

# auto_eval_df = pd.DataFrame(index=[model[:-len('_summary')] for model in models], columns = auto_eval_columns)

# def fill_auto_eval_df(auto_eval_df, mean_values):
#     # Loop through each index in the DataFrame
#     for index in auto_eval_df.index:
#         # Loop through each column in the DataFrame
#         for col in auto_eval_df.columns:
#             # Construct the corresponding column name in mean_values (e.g., 'bart_accuracy', 'bart_overall')
#             column_name = f'{col}_{index}'
#             # Use .get() to safely fetch the value from mean_values, defaulting to None if not found
#             auto_eval_df.loc[index, col] = mean_values.get(column_name+'_summary', None)
    
#     return auto_eval_df

# # Example of usage
# auto_eval_df = fill_auto_eval_df(auto_eval_df, mean_values)

# concatenated_df = pd.concat([human_df, llm_eval_df, auto_eval_df], axis=1)

# column_order = ['human_accuracy', 'human_overall', 'human_coverage',
#        'human_clarity', 'llm_accuracy', 'llm_overall', 'llm_coverage',
#        'llm_clarity', 'rouge1_precision', 'rouge1_recall', 'rouge1_f1',
#        'rouge2_precision', 'rouge2_recall', 'rouge2_f1', 'rougeL_precision',
#        'rougeL_recall', 'rougeL_f1', 'bert_score_precision',
#        'bert_score_recall', 'bert_score_f1', 'summac_zs', 'summac_conv',
#        'summa_qa_prob', 'summa_qa_fscore', 'chrf', 'meteor', 'blanc',
#        'rouge_we_p', 'rouge_we_r', 'rouge_we_f', 'bart_score', 'qa_eval_f1',
#        'qa_eval_em', 'qa_eval_is_answered', 'quest_ex_level_scores','fre_score','dcr_score','bleu_score']

# concatenated_df[column_order].to_excel(r'/raid/huyen/SumDS/cnn_dm/new_scripts/output_auto_evaluation/output_corrected/merged_patent_sum_eval_data.xlsx')
# print('the ned')




import pandas as pd

class PrepareForCorrelation:
    def __init__(self, llm_eval_file, auto_eval_file='', models=["meta-llama/Meta-Llama-3.1-8B-Instruct","Saxo/Linkbricks-Horizon-AI-Avengers-V6-32B","tanliboy/lambda-qwen2.5-14b-dpo-test"]):
        if not auto_eval_file == '':
            self.auto_eval_data = pd.read_excel(auto_eval_file)
        else:
            self.auto_eval_data = pd.DataFrame()
        self.llm_eval_data = pd.read_excel(llm_eval_file)
        self.summary_columns = [col for col in self.llm_eval_data.columns if ((col.endswith('summary')) and not (col.startswith('llm_')))]
        self.metrics = metrices_dict.get(llm_eval_file.split('/')[-1].split('.')[0])
        self.evaluators = (
                    ['human'] if any("human_" in col for col in self.llm_eval_data.columns if col.endswith('_score')) 
                    else ['avg_expert', 'avg_turker'] if any("avg_" in col for col in self.llm_eval_data.columns if col.endswith('_score')) 
                    else [col.replace('_summary','') for col in self.llm_eval_data.columns if col.endswith('_summary') and 'llm_' not in col] # Fallback evaluator if neither condition is met
                )
        self.models = models
        self.llm_eval_cols = [f"{summary_column}_{metric}_{model}" for metric in self.metrics for summary_column in self.summary_columns for model in self.models]
        
        self.human_eval_cols = [f"{evaluator}_{metric}_score" for metric in self.metrics for evaluator in self.evaluators]
        self.auto_eval_cols = [col for col in self.auto_eval_data.columns if 'auto_eval' in col]
        self.aggregated_columns = [f"{aggregation}_{summary_column}_{metric}" for aggregation in ['average','mode'] for metric in self.metrics for summary_column in self.summary_columns]
        # self.auto_eval_cols = self._get_eval_columns(self.auto_eval_data, suffix="_summary")
        
        self.final_cols_data = self._prepare_final_data()

    def _prepare_final_data(self):
        final_data = pd.concat([self.llm_eval_data[self.human_eval_cols + self.llm_eval_cols + self.aggregated_columns], 
                                self.auto_eval_data[self.auto_eval_cols]], axis=1)
        final_data = final_data.T

        filtered_data = final_data[~final_data.index.str.contains('llm_response|explanation', case=False, na=False)]
        self.mean_values = filtered_data.mean(axis=1)
        return final_data

    def _fill_df(self, df, column_names, prefix):
        for index in df.index:
            for metric in self.metrics:
                for model in self.models:
                    if prefix == "human":
                        df.loc[index, f'{prefix}_{metric}_score'] = self.mean_values.get(f'{index}_{metric}_score', None)
                    elif prefix == "llm":
                        df.loc[index, f'{prefix}_{metric}_{model}'] = self.mean_values.get(f'{index}_summary_{metric}_{model}', None)
                    elif prefix == "aggregated":
                        for aggregator in ['average', 'mode']:
                            df.loc[index, f'{aggregator}_{metric}'] = self.mean_values.get(f'{aggregator}_{index}_summary_{metric}', None)
        return df

    def prepare_data(self,output_path):
        """
        Prepare and return the concatenated DataFrame with human, LLM, and Auto evaluation data.
        :return: Concatenated DataFrame containing all processed data
        """
        # Prepare human DataFrame
        human_columns = ['human_'+metric+'_score' for metric in self.metrics]
        human_df = pd.DataFrame(index=[model[:-len('_summary')] for model in self.summary_columns], columns=human_columns)
        human_df = self._fill_df(human_df, human_columns, "human")
        
        # Prepare LLM DataFrame
        llm_eval_columns = ['llm_'+ metric + '_' + model for metric in self.metrics for model in self.models]
        llm_eval_df = pd.DataFrame(index=[model[:-len('_summary')] for model in self.summary_columns], columns=llm_eval_columns)
        llm_eval_df = self._fill_df(llm_eval_df, llm_eval_columns, "llm")
        
        # Prepare Auto Evaluation DataFrame
        auto_eval_columns = list(set([col.replace(f"_{model}", '') for col in self.auto_eval_cols for model in self.models if f"_{model}" in col]))
        auto_eval_df = pd.DataFrame(index=[model[:-len('_summary')] for model in self.summary_columns], columns=auto_eval_columns)
        auto_eval_df = self._fill_df(auto_eval_df, auto_eval_columns, "auto")

        aggregated_columns = [f"{aggregation}_{metric}" for aggregation in ['average','mode'] for metric in self.metrics]
        aggregated_df = pd.DataFrame(index=[model[:-len('_summary')] for model in self.summary_columns], columns=aggregated_columns)
        aggregated_df = self._fill_df(aggregated_df, aggregated_columns, "aggregated")
        
        # Concatenate all DataFrames
        concatenated_df = pd.concat([human_df, llm_eval_df, auto_eval_df, aggregated_df], axis=1)
        concatenated_df.to_excel(output_path)
        return concatenated_df

# auto_eval_file = r''
# llm_eval_file = r'output_multi_agent/output_llm_evaluation/patent_sum_eval_data.xlsx'
# prepare = PrepareForCorrelation(llm_eval_file, auto_eval_file)
# prepare.prepare_data(r'output_multi_agent/output_llm_evaluation/merged_patent_data.xlsx')
