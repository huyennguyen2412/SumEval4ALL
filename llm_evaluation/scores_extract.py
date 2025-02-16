import ast
import pandas as pd
import re

def extract_scores_if_missing(raw_response, missing_scores_columns):
    # If any of the scores are missing, attempt to extract from raw response
    raw_response = raw_response.replace('   ','').replace('\n','')
    missing_scores_dict = {col:None for col in missing_scores_columns}
    raw_response = raw_response.replace('Overall Quality','overall')
    # if isinstance(raw_responses,str):
    #     raw_responses = ast.literal_eval(raw_responses)
        

    for col in missing_scores_columns:
        # if not :
        #     # Get the indices of None values in the cell
        #     none_indices = [i for i, value in enumerate(row[col]) if value is None]
            
        #     # Get corresponding raw_responses for those indices
        #     responses_with_none = [raw_responses[i] for i in none_indices]

        # for index,raw_response in enumerate(responses_with_none):
        # response_scores = {col: None for col in columns if col != 'explanation'}
        # explanation = None

        if pd.isna(raw_response):
            raw_response = ''

        # for col in columns:

        if col != 'explanation' and missing_scores_dict[col] is None:
            match = re.search(fr"'{'_'.join(col.split('_')[-2:])}':\s*(\d+\.?\d*)", raw_response)
            if match:
                missing_scores_dict[col] = (float(match.group(1)))

        if col != 'explanation' and missing_scores_dict[col] is None:
            match = re.search(fr"'{' '.join(col.split('_')[-2:])}'\s*:\s*(\d+\.?\d*)", raw_response)
            if match:
                missing_scores_dict[col] = float(match.group(1))

        # Try alternative pattern (e.g., **Coherence:** 4 or **Coherence:** 4/5)
        # for col in columns:
        if col != 'explanation' and missing_scores_dict[col] is None:
            match = re.search(fr"\*\*{' '.join(col.split('_')[-2:]).capitalize()}:\*\*\s*(\d+\.?\d*)/?\d*", raw_response, re.IGNORECASE)
            if match:
                missing_scores_dict[col] = float(match.group(1))

        if col != 'explanation' and missing_scores_dict[col] is None:
            match = re.search(fr"{' '.join(col.split('_')[-2:]).capitalize()}:\s*(\d+\.?\d*)/?\d*", raw_response, re.IGNORECASE)
            if match:
                missing_scores_dict[col] = float(match.group(1))


        # Extract explanation
        if col == 'explanation':
            explanation_match = re.search(r"'explanation':\s*(.*)", raw_response, re.DOTALL)
            if not explanation_match:
                explanation_match = re.search(r"'explaining':\s*(.*)", raw_response, re.DOTALL)
            if not explanation_match:
                explanation_match = re.search(r"\*\*Explanation:\*\*\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL)
            if not explanation_match:
                explanation_match = re.search(r"Explanation:\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL)
            
            missing_scores_dict[col] = explanation_match.group(1).strip() if explanation_match else None
        
        # Append extracted values for the current response
        # for col in columns:
        # if col != 'explanation':
        #     row[col][none_indices[index]] = response_scores[col]
        # else:
        #     row['explanation'][none_indices[index]] = explanation

    return missing_scores_dict



# df = pd.read_excel('output_data/arxiv-gov_report_data_split_sentence.xlsx')
# columns = ['relevance_sentence_level','factual_consistency_sentence_level','explanation']
# # Apply the updated function to extract missing scores
# # df_updated = df.apply(extract_scores_if_missing, axis=1)
# df_updated = df.apply(lambda row: extract_scores_if_missing(row, columns), axis=1)

# # Save the updated DataFrame to a new CSV file
# output_path = 'output_data/arxiv-gov_report_data_split_sentence.xlsx'
# df_updated.to_excel(output_path, index=False)

# # Output the new CSV file path
# print(f"Updated CSV saved to: {output_path}")
