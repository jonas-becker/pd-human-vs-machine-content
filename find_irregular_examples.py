import os
import pandas as pd
from tqdm import tqdm
from setup import *

'''
Finds examples with very low results for a specific methods that others did not perform as bad in
Requires the get_examples.py being processed beforehand as it takes the json example files from the files generated with the script
'''

print("Finding irregular examples that are unique per dataset...")

for file in os.listdir(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection")):
    if "originals" in file or "random" in file:
        continue
    print(f"Handling file {file}")
    if file.split("_")[1] == "semantic" or file.split("_")[1] == "tfidf":
        method = file.split("_")[1]+"_"+file.split("_")[2]
    else:
        method = file.split("_")[1]
    dataset = file.split("_")[0]

    df = pd.read_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection", file), orient = "index")
    # only keep examples with lowest confidence 
    df = df.sort_values(method, ascending=True).head(int(EXAMPLE_AMOUNT/4))

    other_df = pd.DataFrame()

    for other_file in os.listdir(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection")):
        if "originals" in other_file or dataset not in other_file or method in other_file:
            continue
        print(f"Handling other file {other_file}")
        if other_file is not file:
            other_df = pd.concat([other_df, pd.read_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection", other_file), orient = "index").sort_values(method, ascending=True).head(int(EXAMPLE_AMOUNT/2))])

    print(len(df))
    print(len(other_df))
    df = df[~df[PAIR_ID].isin(other_df[PAIR_ID])].sort_values(method, ascending=True)
    df.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "irregulars", dataset+"_"+method+".json"), orient = "index", index = True, indent = 4)
    print(f"A total of {str(len(df))} entries are unique for the {method} method and {dataset} dataset.")

print("Done.")
        