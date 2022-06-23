import os
import pandas as pd
from setup import *

for dataset_file in os.listdir(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER)):
    df_examples = pd.DataFrame()

    print("Processing " + dataset_file)
    df = pd.read_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, dataset_file), orient = "index")
    #lowest (paraphrase)
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(COSINE_DISTANCE, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(5) ] )
    #highest (paraphrase)
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(COSINE_DISTANCE, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(5) ] )

    #Output data to json format
    df_examples.reset_index(inplace=True, drop=True)
    df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, str(dataset_file).split("_")[0]+"_paraphrases.json" ), orient = "index", index = True, indent = 4)
    df_examples = pd.DataFrame()

    # highest (non-paraphrase)
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(COSINE_DISTANCE, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(5) ] )
    #lowest (non-paraphrase)
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(COSINE_DISTANCE, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(5) ] )

    #Output data to json format
    df_examples.reset_index(inplace=True, drop=True)
    df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, str(dataset_file).split("_")[0]+"_originals.json" ), orient = "index", index = True, indent = 4)
    
print("Done.")