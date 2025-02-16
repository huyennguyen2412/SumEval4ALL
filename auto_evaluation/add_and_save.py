import pandas as pd

def concat_and_save_df(input_data,all_scores,out_file):
    column_order = list(input_data.columns) + list(all_scores.columns)
    out_df = input_data.combine_first(all_scores)[list(dict.fromkeys(column_order))]
    if out_file.split('.')[-1] == 'xlsx':
        out_df.to_excel(out_file,index = False)
    elif out_file.split('.')[-1] == 'csv':
        out_df.to_csv(out_file, encoding='utf-8',index = False)
    return out_df

def update_and_save_df(out_df, row_index, column_name, result, out_file):
    out_df.at[row_index, column_name] = result
    if out_file.split('.')[-1] == 'xlsx':
        out_df.to_excel(out_file, index=False)
    elif out_file.split('.')[-1] == 'csv':
        out_df.to_csv(out_file, encoding='utf-8', index=False)
    return out_df