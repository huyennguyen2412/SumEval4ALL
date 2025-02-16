import os
from llm_evaluation.generate_summary import GenSummary
from llm_evaluation.llm_evaluation import EvaluateSummary
from llm_evaluation.final_summary import ImproveSummary

tasks_to_perform = ['generate-summary', 'evaluate-summary', 'improve-evaluate-summary', 'multi-agent-evaluation']

for task in tasks_to_perform:
    if task == 'generate-summary':
        input_file_paths = [r'input/eval_datasets/arxiv_data.xlsx',r'input/eval_datasets/gov_report_data.xlsx',r'input/eval_datasets/patent_sum_eval_data.xlsx',r'input/eval_datasets/qags_cnn_dm_data.csv',r'input/eval_datasets/qags_x_sum_data.csv',r'input/eval_datasets/summ_eval_data.csv',r'input/eval_datasets/tldr_data.xlsx']
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" #"meta-llama/Llama-2-7b-chat-hf" # "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        article_column = 'article_content'
        summary_generator = GenSummary(input_file_paths, model_id = model_id, article_column = article_column)
        
        summary_generator.generate_summaries()
    
    if task == 'evaluate-summary':
        input_file_paths = [r'output_llm_summary/arxiv_data.xlsx',r'output_llm_summary/gov_report_data.xlsx',r'output_llm_summary/patent_sum_eval_data.xlsx',r'output_llm_summary/qags_cnn_dm_data.csv',r'output_llm_summary/qags_x_sum_data.csv',r'output_llm_summary/summ_eval_data.csv',r'output_llm_summary/tldr_data.xlsx']
        # model_ids = ["meta-llama/Meta-Llama-3.1-8B-Instruct"] #"meta-llama/Llama-2-7b-chat-hf" # "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3.1-8B-Instruct" 
        model_ids = ["mistralai/Mistral-7B-v0.1","NousResearch/Hermes-2-Theta-Llama-3-8B","microsoft/phi-4","arcee-ai/Arcee-Nova","Qwen/Qwen2-7B-Instruct","allenai/Llama-3.1-Tulu-3-8B"]
        # model_ids = ["meta-llama/Meta-Llama-3.1-8B-Instruct","Saxo/Linkbricks-Horizon-AI-Avengers-V6-32B","tanliboy/lambda-qwen2.5-14b-dpo-test"] ### for multi-agent evaluation
        article_column = 'article_content'
        summary_columns = ['llm_summary']
        summary_sentence_column = 'summary_sentences'
        summary_evaluator = EvaluateSummary(input_file_paths, article_column = article_column, summary_columns = summary_columns, 
                                            summary_sentence_column =  summary_sentence_column, levels = 'summary-level',
                                            model_ids = model_ids)
        
        summary_evaluator.evaluate_summaries()
    
    if task == 'improve-evaluate-summary':
        input_file_paths = [r'output_llm_evaluation/arxiv_data.xlsx',r'output_llm_evaluation/gov_report_data.xlsx',r'output_llm_evaluation/patent_sum_eval_data.xlsx',r'output_llm_evaluation/qags_cnn_dm_data.csv',r'output_llm_evaluation/qags_x_sum_data.csv',r'output_llm_evaluation/summ_eval_data.csv',r'output_llm_evaluation/tldr_data.xlsx']
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" #"meta-llama/Llama-2-7b-chat-hf" # "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3.1-8B-Instruct"
        article_column = 'article_content'
        summary_column = 'llm_summary'
        explanation_column = summary_column + '_' + 'explanation'
        metric_to_compare = 'overall'
        
        summary_generator = ImproveSummary(input_file_paths, model_id = model_id, article_column = article_column, 
                                           summary_column = summary_column, explanation_column = explanation_column, 
                                           metric_to_compare = metric_to_compare, output_dir='output_multi_model')
        
        summary_generator.improve_summaries()

        improved_summary_file_path = [os.path.join('output_improved_llm_summary',os.path.split(file)[1]) for file in input_file_paths]
        model_ids = ["meta-llama/Meta-Llama-3.1-8B-Instruct"]
        summary_evaluator = EvaluateSummary(improved_summary_file_path, article_column = article_column, 
                                            summary_columns = ['improved_llm_summary'],
                                            is_improved = True,
                                            model_ids = model_ids)
        
        summary_evaluator.evaluate_summaries()
