import os
from auto_evaluation.prepare_correlation import RunCorrelation
from auto_evaluation.eval import AutoEvaluation

tasks_to_perform = ['auto-evaluation', 'correlation']

for task in tasks_to_perform:
    if task == "auto-evaluation":
        ### Make this in for loop and remove the need to give output file path
        evaluations_to_perform = ['rouge_score','bert_score','summa_c_scores','summa_qa_scores','chrf_score','meteor_score','blanc_score','rouge_we_score','supert_score','bart_score','mover_score','qa_eval','quest_eval','feqa','bleu_score','fre_score','dcr_score']
        
        input_file_paths = [r'input/input/eval_datasets/arxiv_data.xlsx',r'input/input/eval_datasets/gov_report_data.xlsx',r'input/input/eval_datasets/qags_cnn_dm_data.xlsx',r'input/input/eval_datasets/qags_x_sum_data.xlsx',r'input/input/eval_datasets/sum_eval_evaluation_scores.csv',r'input/input/eval_datasets/tldr_data.xlsx',r'input/input/eval_datasets/patent_sum_eval_data.xlsx']
        # input_file_paths = [r'output_auto_evaluation/output_corrected/tldr_evaluation_scores.csv']
        # input_file_paths = [r'output_auto_evaluation/output_corrected/patent_sum_eval_data.xlsx']
        article_column = 'article_content'
        summary_columns = ['ummary']
        # summary_columns = ['bart_summary','gpt35_summary','hupd-t5-base_summary','long-t5-tglobal-base-16384-book-summary-2_summary','xl_net_summary']
        # evaluations_to_perform = ['bart_score']
        # evaluations_to_perform = ['rouge_score','bert_score']
        auto_eval_runner = AutoEvaluation(input_file_paths = input_file_paths, article_column = article_column, summary_columns = summary_columns)
        auto_eval_runner.run_auto_evaluation(evaluations_to_perform)

    if task == 'correlation':
        input_file_paths = [r"output_auto_evaluation/output_corrected/merged_patent_sum_eval_data.xlsx"]
        evaluators = ['human']
        metrics_to_compare = ['human_vs_llm_eval','human_vs_auto_eval','all']
        human_metrices = ['accuracy','overall','coverage','clarity']
        correlation_methods = ['pearson', 'kendall', 'spearman']
        model_column = 'model'
        
        correlation_runner = RunCorrelation(input_file_paths = input_file_paths, evaluators = evaluators, model_column = model_column,
        correlation_methods = correlation_methods, metrics_to_compare = metrics_to_compare, 
        human_metrices = human_metrices, agg_human_metrices = False, group_by_model = False)
        
        correlation_runner.run_all_correlations()