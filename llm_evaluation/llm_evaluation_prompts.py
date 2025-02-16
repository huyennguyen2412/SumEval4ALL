prompt_dict = {

"qags_cnn_dm_data_summary-level": """
                                I would like you to assess the factual consistency of summaries. To evaluate the summaries, please follow the instructions below:

                                - In this task, you will evaluate the factual consistency of summaries provided. The task is to assess whether the entire summary is factually accurate based on the information available in the source document. Summaries may contain content directly copied from the source document or combine information from multiple sections.

                                    Key points to consider:
                                    - Some summaries might mix information from different parts of the article, which could lead to inaccuracies or unsupported claims.
                                    - If a part of the summary directly matches content from the article, treat it as factually consistent.
                                    - If the summary adds, contradicts, or fabricates information, it is not factually supported.
                                    - Some article sentences, such as "Scroll down for video," may seem out of place. If the summary includes one of these, treat it as factually supported.

                                - **Scoring**: Provide a binary score based on the summary’s overall factual consistency:
                                    - **1 (Supported)**: The summary is fully factually consistent with the source document.
                                    - **0 (Not Supported)**: The summary contains inaccuracies, contradictions, or unsupported claims.

                                - **Explanation**: For each score, provide a brief explanation:
                                    - If the score is **1**, explain how the summary aligns with the article’s content.
                                    - If the score is **0**, identify inaccuracies, contradictions, or unsupported statements.

                                Output format:
                                {
                                    'factual_consistency': binary_score,
                                    'explanation': "Your explanation here"
                                }

                                Remember to give the output in json format.
                                """
,

"qags_cnn_dm_data_summary-sentence-level": """
                                I would like you to assess the factual consistency of individual sentences from summaries. To evaluate the sentence, please follow the instructions below:

                                - In this task, you will evaluate whether a particular summary sentence is factually accurate based on the information available in the source document. The sentence may either be directly copied from the article or combine information from different sections of the article.

                                    Key points to consider:
                                    - Some sentences might mix information from different parts of the article, which could result in inaccuracies or unsupported claims.
                                    - If the sentence directly matches content from the article, treat it as factually consistent.
                                    - If the sentence adds, contradicts, or fabricates information, it is not factually supported.
                                    - Some article sentences, such as "Scroll down for video," may seem out of place. If the sentence includes such content, treat it as factually supported.

                                - **Scoring**: Provide a binary score based on the factual consistency of the sentence:
                                    - **1 (Supported)**: The sentence is fully factually consistent with the source article.
                                    - **0 (Not Supported)**: The sentence contains inaccuracies, contradictions, or unsupported information.

                                - **Explanation**: For each score, provide a brief explanation:
                                    - If the score is **1**, explain how the sentence aligns with the article’s content.
                                    - If the score is **0**, identify specific inaccuracies, contradictions, or unsupported statements within the sentence.

                                Output format:
                                {
                                    'factual_consistency': binary_score,
                                    'explanation': "Your explanation here"
                                }
                                
                                Remember to give the output in json format.
                                """
,

"qags_x_sum_data_summary-level": """
                                I would like you to assess the factual consistency of summaries. To evaluate, please follow the instructions below:

                                - The task is to determine if the summary is factually correct given the contents of the article. **All parts of the summary must be stated or implied by the article to be considered correct.**

                                    For example:
                                    - If the summary discusses "John Smith" but the article only talks about "Mr. Smith," the fact that the person's first name is John is NOT supported.
                                    - If the summary mentions a 15-year-old girl but the article only discusses a young girl, the fact that she is 15 is NOT supported.

                                - Verifying a summary will often require combining facts from many different parts of the article, so read the entire article closely.
                                - If the summary directly copies content from the article, you should mark it as supported.
                                - If the summary contains information that contradicts, fabricates, or adds to what is in the article, it is NOT supported.
                                - If the summary contains claims or details (e.g., names, numbers, ages, locations) not supported by the article, it is NOT factually consistent.

                                **Scoring**: Provide a binary score for the summary’s factual consistency:
                                    - **1 (Supported)**: The summary is factually consistent with the source document.
                                    - **0 (Not Supported)**: The summary contains inaccuracies, contradictions, or unsupported claims.

                                **Explanation**: For each score, provide a detailed explanation:
                                    - If the score is **1**, explain how the summary aligns with the article's content.
                                    - If the score is **0**, identify specific inaccuracies, contradictions, or unsupported statements within the summary.

                                Output format:
                                {
                                    'factual_consistency': binary_score,
                                    'explanation': "Your explanation here"
                                }
                                
                                Remember to give the output in json format.
                                """
,

"qags_x_sum_data_summary-sentence-level":"""
                                I would like you to assess the factual consistency of individual sentences from summaries. To evaluate, please follow the instructions below:

                                - The task is to determine if the sentences are factually correct given the contents of the article. **All parts of the sentence must be stated or implied by the article to be considered correct.** 

                                    For example:
                                    - If the sentence discusses "John Smith" but the article only talks about "Mr. Smith," the fact that the person's first name is John is NOT supported.
                                    - If the sentence mentions a 15-year-old girl but the article only discusses a young girl, the fact that she is 15 is NOT supported.

                                - Verifying a sentence will often require combining facts from many different parts of the article, so read the entire article closely. 
                                - If the sentence directly copies the article, you should mark it as supported.
                                - If the sentence doesn't make sense, you should mark it as not supported.

                                **Scoring**: Provide a binary score for the sentence’s factual consistency:
                                    - **1 (Supported)**: The sentence is factually consistent with the source document.
                                    - **0 (Not Supported)**: The sentence contains inaccuracies, contradictions, or unsupported information.

                                **Explanation**: For each score, provide a brief explanation:
                                    - If the score is **1**, explain how the sentence aligns with the source document.
                                    - If the score is **0**, identify specific inaccuracies, contradictions, or unsupported claims.

                                Output format:
                                {
                                    'factual_consistency': binary_score,
                                    'explanation': "Your explanation here"
                                }

                                Remember to give the output in json format.
                                """
,

"patent_sum_eval_data_summary-level":"""
                                I would like you to assess the quality of summaries. To evaluate the quality of the
                                summaries, you need to consider the following dimensions:
                                
                                • Clarity: Is the summary reader-friendly? Does it express ideas clearly?
                                • Accuracy: Does the summary contain the same information as the original
                                document?
                                • Coverage: How well does the summary cover the important information in the
                                original document?
                                • Overall quality: How good is the summary overall at representing the original
                                document?
                                
                                Now given below a original document, and its summary. On scale of 1-5, what are the scores of 
                                Clarity, Accuracy, Coverage, and Overall quality of the summary?
                                Provide one to two short sentences to explain why you gave that rating for each dimension.
                                
                                The output must follow the following Python dictionary format: 
                                
                                {'clarity': clarity_score, 'accuracy': accuracy_score, 'coverage': coverage_score, 
                                'overall': overall_quality_score, 
                                'explanation':{'clarity': a_explanation, 'accuracy': a_explanation, 'coverage': a_explanation, 'overall': a_explanation}}
                                """
,

"tldr_data_summary-level":"""
                            I would like you to assess the quality of summaries. To evaluate the quality of the summaries, please consider the following dimensions:

                            - **Coherence**: How coherent is the summary on its own? Is it easy to understand and free of English errors? A summary is coherent if it’s clear and makes sense when read by itself.
                                - Score of 1: The summary is impossible to understand.
                                - Score of 4: The summary has mistakes or confusing phrasing that make it a bit hard to understand.
                                - Score of 7: The summary is perfectly clear.

                            - **Accuracy**: Does the factual information in the summary accurately match the original article? A summary is accurate if it doesn’t introduce any facts not present in the post and doesn't contradict anything.
                                - Score of 1: The summary is completely wrong, made up, or exactly contradicts what is written in the post.
                                - Score of 4: The summary says at least one substantial thing that is not mentioned in the post, or that contradicts something in the post.
                                - Score of 5: The summary says anything, no matter how small, that is not mentioned in the post, or that contradicts something in the post.
                                - Score of 7: The summary has no incorrect statements or misleading implications.

                            - **Coverage**: How well does the summary cover the important information in the post? A summary has good coverage if it includes the key points required to understand the situation described in the post.
                                - Score of 1: The summary contains no information relevant to the post.
                                - Score of 4: The summary is missing at least 1 important piece of information required to understand the situation.
                                - Score of 7: The summary covers all of the important information required to understand the situation.

                            - **Overall Quality**: How good is the summary overall at representing the post? This encompasses all of the above axes of quality, as well as other aspects you think are important.
                                - Score of 1: The summary is terrible.
                                - Score of 4: The summary is an okay representation of the post, but could be significantly improved.
                                - Score of 7: The summary is an excellent representation of the post.

                            Provide scores (1-7) for Coherence, Accuracy, Coverage, and Overall Quality, along with short explanations for each score. The explanation should describe how well the summary performs in each of the respective areas based on the rubric criteria.

                            Output format:
                            {
                                'coherence': score,
                                'accuracy': score,
                                'coverage': score,
                                'overall': score,
                                'explanation': {
                                    'coherence': explanation,
                                    'accuracy': explanation,
                                    'coverage': explanation,
                                    'overall': explanation
                                }
                            }
""",

"arxiv_data_summary-level":"""I would like you to assess the quality of summaries. To evaluate the quality of the summaries, please consider the following dimensions:

- **Relevance**: Does the summary contain the main ideas of the source document? A summary is relevant if it includes the key concepts of the original document without deviating from the core ideas.
    - Score of 1: The summary contains the main ideas and is aligned with the source document.
    - Score of 0: The summary is missing important information or introduces incorrect ideas that are not in the source document.

- **Factual Consistency**: Is the summary factually consistent with the source? A summary is factually consistent if there are no errors in terms of the details presented in the source document, such as incorrect entities, predicates, or events.
    - Score of 1: The summary is factually consistent with the source, with no errors.
    - Score of 0: The summary contains factual errors, such as incorrect entities, incorrect relationships, or misleading information.

Please evaluate each summary for relevance and factual consistency, using a **binary scale** of 1 (relevant/factual) or 0 (not relevant/factual). For each summary, provide the following:

- **Relevance**: Score (1 or 0) based on how well the summary captures the main ideas of the source document.
- **Factual Consistency**: Score (1 or 0) based on how consistent the summary is with the factual details in the source document.

Output format:

{
    'relevance': score,
    'factual_consistency': score,
    'explanation': {
        'relevance': explanation of why the summary is or is not relevant,
        'factual_consistency': explanation of why the summary is or is not factually consistent
    }
}
""",

"gov_report_data_summary-level":"""I would like you to assess the quality of summaries. To evaluate the quality of the summaries, please consider the following dimensions:

- **Relevance**: Does the summary contain the main ideas of the source document? A summary is relevant if it includes the key concepts of the original document without deviating from the core ideas.
    - Score of 1: The summary contains the main ideas and is aligned with the source document.
    - Score of 0: The summary is missing important information or introduces incorrect ideas that are not in the source document.

- **Factual Consistency**: Is the summary factually consistent with the source? A summary is factually consistent if there are no errors in terms of the details presented in the source document, such as incorrect entities, predicates, or events.
    - Score of 1: The summary is factually consistent with the source, with no errors.
    - Score of 0: The summary contains factual errors, such as incorrect entities, incorrect relationships, or misleading information.

Please evaluate each summary for relevance and factual consistency, using a **binary scale** of 1 (relevant/factual) or 0 (not relevant/factual). For each summary, provide the following:

- **Relevance**: Score (1 or 0) based on how well the summary captures the main ideas of the source document.
- **Factual Consistency**: Score (1 or 0) based on how consistent the summary is with the factual details in the source document.

Output format:
{
    'relevance': score,
    'factual_consistency': score,
    'explanation': {
        'relevance': explanation of why the summary is or is not relevant,
        'factual_consistency': explanation of why the summary is or is not factually consistent
    }
}
""",

"arxiv_data_summary-sentence-level":"""I would like you to assess the quality of individual sentences from summaries. To evaluate the quality, please consider the following dimensions:

- **Relevance**: Does the sentence contain the main ideas of the source document? A sentence is relevant if it includes key concepts of the original document without deviating from the core ideas.
    - Score of 1: The sentence captures the main ideas and is aligned with the source document.
    - Score of 0: The sentence is missing important information or introduces incorrect ideas that are not in the source document.

- **Factual Consistency**: Is the sentence factually consistent with the source? A sentence is factually consistent if there are no errors in terms of the details presented in the source document, such as incorrect entities, predicates, or events.
    - Score of 1: The sentence is factually consistent with the source, with no errors.
    - Score of 0: The sentence contains factual errors, such as incorrect entities, incorrect relationships, or misleading information.

Please evaluate each sentence for relevance and factual consistency, using a **binary scale** of 1 (relevant/factual) or 0 (not relevant/factual). For each sentence, provide the following:

- **Relevance**: Score (1 or 0) based on how well the sentence captures the main ideas of the source document.
- **Factual Consistency**: Score (1 or 0) based on how consistent the sentence is with the factual details in the source document.

Output format:
{
    'relevance': score,
    'factual_consistency': score,
    'explanation': {
        'relevance': explanation of why the sentence is or is not relevant,
        'factual_consistency': explanation of why the sentence is or is not factually consistent
    }
}
""",

"gov_report_data_summary-sentence-level":"""I would like you to assess the quality of individual sentences from summaries. To evaluate the quality, please consider the following dimensions:

- **Relevance**: Does the sentence contain the main ideas of the source document? A sentence is relevant if it includes key concepts of the original document without deviating from the core ideas.
    - Score of 1: The sentence captures the main ideas and is aligned with the source document.
    - Score of 0: The sentence is missing important information or introduces incorrect ideas that are not in the source document.

- **Factual Consistency**: Is the sentence factually consistent with the source? A sentence is factually consistent if there are no errors in terms of the details presented in the source document, such as incorrect entities, predicates, or events.
    - Score of 1: The sentence is factually consistent with the source, with no errors.
    - Score of 0: The sentence contains factual errors, such as incorrect entities, incorrect relationships, or misleading information.

Please evaluate each sentence for relevance and factual consistency, using a **binary scale** of 1 (relevant/factual) or 0 (not relevant/factual). For each sentence, provide the following:

- **Relevance**: Score (1 or 0) based on how well the sentence captures the main ideas of the source document.
- **Factual Consistency**: Score (1 or 0) based on how consistent the sentence is with the factual details in the source document.

Output format:
{
    'relevance': score,
    'factual_consistency': score,
    'explanation': {
        'relevance': explanation of why the sentence is or is not relevant,
        'factual_consistency': explanation of why the sentence is or is not factually consistent
    }
}
"""
,

"summ_eval_data_summary-level": """ I would like you to assess the quality of summaries. To evaluate the quality of the summaries, you need to consider the following dimensions:

Coherence: The rating measures the quality of all sentences collectively, their fit together, and how naturally they sound. Consider the quality of the summary as a whole.
Consistency: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary reproduces all facts accurately and does not make up untrue information.
Fluency: This rating measures the quality of individual sentences. Are they well-written and grammatically correct? Consider the quality of individual sentences.
Relevance: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.

Now, given below is the original document and its summary. On a scale of 1–5, what are the scores for Coherence, Consistency, Fluency, and Relevance of the summary?

Provide one to two short sentences to explain why you gave that rating for each dimension. The output must follow the following Python dictionary format:
{
    'coherence': coherence_score, 
    'consistency': consistency_score, 
    'fluency': fluency_score, 
    'relevance': relevance_score, 
    'explanation': {
        'coherence': a_explanation, 
        'consistency': a_explanation, 
        'fluency': a_explanation, 
        'relevance': a_explanation
    }
}

""",

"general_summary-level":"""
                        I would like you to assess the quality of summaries. To evaluate the quality of the
                        summaries, you need to consider the following dimensions:
                            • Clarity: Is the summary reader-friendly? Does it express ideas clearly?
                            • Accuracy: Does the summary contain the same information as the original
                            document?
                            • Coverage: How well does the summary cover the important information in the
                            original document?
                            • Overall quality: How good is the summary overall at representing the original
                        document?

                        **Rubric**
                        *Clarity*:
                            • Score of 1 (Poor): The summary is extremely unclear and difficult to understand. It's filled with grammatical errors, ambiguous phrasing, and confusing language.
                            • Score of 2 (Fair): The summary is somewhat clear but still contains instances of unclear or awkward language, requiring effort from readers to understand. It needs improvement for better understanding.
                            • Score of 3 (Adequate):The summary is generally clear and understandable, although it could be further improved for better understanding.
                            • Score of 4 (Good):The summary is clear and easily understandable, effectively conveying ideas with well-structured sentences and clear language.
                            • Score of 5 (Excellent):The summary is exceptionally clear, concise, and engaging, demonstrating outstanding readability and expression of ideas.
                        *Accuracy*:
                            • Score of 1 (Poor): The summary contains numerous factual errors and inaccuracies compared to the original document. It misrepresents key details or presents information that is highly misleading or inaccurate compared to the original document.
                            • Score of 2 (Fair):The summary has noticeable inaccuracies or inconsistencies compared to the original document. It captures some information correctly but also includes several errors, affecting the overall accuracy.
                            • Score of 3 (Adequate):The summary represents most information from the original document correctly. There might be a few minor inaccuracies, but they don't significantly affect the overall understanding.
                            • Score of 4 (Good):The summary is highly accurate, closely mirroring the information presented in the original document.
                            • Score of 5 (Excellent):The summary is highly precise and accurate, faithfully representing the original document's information.
                        *Coverage*:
                            • Score of 1 (Poor): The summary misses most essential information from the original document.
                            • Score of 2 (Fair):The summary covers a moderate amount of important information but lacks depth or misses key aspects of the original document.
                            • Score of 3 (Adequate):The summary adequately covers important information from the original document. It captures most key points but may lack some minor details, providing a satisfactory representation.
                            • Score of 4 (Good):The summary covers important information from the original document. It captures the majority of key points and details, offering a thorough representation.
                            • Score of 5 (Excellent):The summary extensively covers all crucial information from the original document, encompassing every significant point and detail effectively. It provides an outstandingly comprehensive and complete representation.
                        *Overall Quality*:
                            • Score of 1 (Poor): The summary is extremely terrible.
                            • Score of 2 (Fair):The summary has many quality issues, and badly represent the main points of the original document.
                            • Score of 3 (Adequate):The summary is an okay representation of the original document, but could be significantly improved
                            • Score of 4 (Good):The summary effectively captures the key elements of the original document but might have minor areas for improvement or refinement.
                            • Score of 5 (Excellent):The summary is an excellent representation of the original document.

                        Now given below a original document, and its summary. On scale of 1-5, what are the scores of 
                        Clarity, Accuracy, Coverage, and Overall quality of the summary?
                        For each dimension, explain briefly why you gave that rating, and explicitly point out what information is accurate (for Accuracy) or missing (for Coverage) if Accuracy and Coverage are low-score.
                        
                        The output must strictly follow the following Python dictionary format. Do not include anything else: 
                        
                        {'clarity': clarity_score, 'accuracy': accuracy_score, 'coverage': coverage_score, 
                        'overall': overall_quality_score, 
                        'explanation':{'clarity': a_explanation, 'accuracy': a_explanation, 'coverage': a_explanation, 'overall': a_explanation}}
                        """

}
