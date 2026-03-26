# RUN from root dir

import pandas as pd 
import json
import os 
import sys 
import numpy as np 
import ast
import re

MAPPING = {
    1.0: 2, 
    2.0: 1, 
    3.0: 0, 
    4.0: -1, 
    5.0: -2,
    'not_sure': -3, 
    0.0: -3, 
    -3.0: -3
}

def get_config_from_id(id):
    try:
        # pattern = r'c_[^_]+(?:_[^_]+)*?(?=_dim)'
        pattern = r'c_[^_]+(?:_[^_]+)*?(?=(?:_dim|_feedback))'
        return re.search(pattern, id).group(0)
    except Exception as e: 
        print(e, id)
        print(re.search(pattern, id))

def get_metric_from_id(id):
    try:
        return id.split("_")[8]
    except Exception as e: 
        # feedback
        return None

def parse_json(flatList, annotator='Comet [AI]', parsed_queries = []):
    results = []

    METRICS = ['relevance', 'diversity', 'popularity', 'sustainability']

    path_to_qs = f"data/conv-trs/ecir-2026/selected_queries/filtered_queries.json"
    configs = pd.read_json(path_to_qs)

    temp_result = {}
    prev = None

    # sort dictionary 
    data_sorted = dict(sorted(flatList.items()))
    
    for key, value in data_sorted.items(): 
        if 'ranking' not in key: 
            continue 
        
        qid = get_config_from_id(key)
        
        if qid in parsed_queries: 
            # was already processed at a previous run 
            print(f"{qid} already processed for {annotator}, moving to the next query")
            continue

        if len(parsed_queries) == 0: 
            # start with the processing
            print(f"Starting the parser for qid: {qid}")
            prev = qid 
            parsed_queries.append(qid)
            temp_result['query_id'] = qid 
            temp_result['query'] = configs[configs['query_id'] == qid]['query_text'].values[0]
            temp_result['annotator'] = annotator
        
        if qid != prev: 
            # query has changed, update results 
            results.append(temp_result)
            print(f"Parsed. Appending dict for qid {prev}")

            prev = qid 
            temp_result = {}
            temp_result['query_id'] = qid 
            temp_result['query'] = configs[configs['query_id'] == qid]['query_text'].values[0]
            temp_result['annotator'] = annotator

        # still the same query
        if 'feedback' in key: 
            temp_result['feedback'] = value 
        else:  
            metric = get_metric_from_id(key)
            if 'not_sure' in key: 
                temp_result[metric] = MAPPING['not_sure']
            else: 
                temp_result[metric] = MAPPING[float(value)]

            print(f"processed metric: {metric} for qid: {qid}")

    print(f"Parsed. Appending dict for {qid}")
    results.append(temp_result)

    return results 

# For other annotators 

def run(annotator_name):
    # access current human_eval_df 
    path_to_survey = "data/conv-trs/ecir-2026/human-eval"

    try: 
        df = pd.read_csv(f"{path_to_survey}/survey_parsed.csv")
    except Exception as e: 
        # CSV doesn't exist - need to run the entire notebook
        print("No CSV found. Please parse the files from scratch!")
        return None
    
    if annotator_name in df['annotator'].unique(): 
        parsed_queries = df[df['annotator'] == annotator_name]['query_id'].tolist()
    else: 
        parsed_queries = []
    
    with open(f"{path_to_survey}/surveys_collection_raw.json", "r") as fp: 
        responses = json.load(fp)
    
    data = None
    for item in responses: 
        if item['prolificId'] == annotator_name: 
            data = item['flatAnswers']
            break 
    
    if not data: 
        print("Annotator not found!")
        return None
    
    res = parse_json(data, annotator_name, parsed_queries)
    res_df = pd.DataFrame(res)
    new_df = pd.concat([df, res_df])

    sorted_new = new_df.sort_values(by=['query_id']).reset_index(drop=True)
    print(f"Parsed data for {annotator_name}. Storing CSV")
    sorted_new.to_csv(f"{path_to_survey}/survey_parsed.csv", index=False)

if __name__ == "__main__":
    run(annotator_name="Yas")
    # run(annotator_name="Dana Marti")
    # run(annotator_name="Tejas Srinivasan")