import os
import pandas as pd
from setup import *
from sklearn.utils import shuffle

for dataset_file in os.listdir(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER)):
    df_examples = pd.DataFrame()

    print("Processing " + dataset_file)
    df = pd.read_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, dataset_file), orient = "index")
    #filter out 1.0 distances (exactly the same texts)
    df = df[df[COSINE_DISTANCE] != 1.0]

    #lowest (paraphrase)
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(COSINE_DISTANCE, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )
    #highest (paraphrase)
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(COSINE_DISTANCE, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )

    #Output data to json format
    df_examples.reset_index(inplace=True, drop=True)
    df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, str(dataset_file).split("_")[0]+"_top_paraphrases.json" ), orient = "index", index = True, indent = 4)
    df_examples = pd.DataFrame()

    # highest (non-paraphrase)
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(COSINE_DISTANCE, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )
    #lowest (non-paraphrase)
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(COSINE_DISTANCE, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )

    #Output data to json format
    df_examples.reset_index(inplace=True, drop=True)
    df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, str(dataset_file).split("_")[0]+"_top_originals.json" ), orient = "index", index = True, indent = 4)
    df_examples = pd.DataFrame()

    # random pairs (paraphrase)

    df = shuffle(df)    #shuffle the df
    # randoms original pairs
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].head(EXAMPLE_AMOUNT) ] )
    df_examples.reset_index(inplace=True, drop=True)
    df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, str(dataset_file).split("_")[0]+"_random_originals.json" ), orient = "index", index = True, indent = 4)    
    df_examples = pd.DataFrame()

    # randoms paraphrased pairs
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].head(EXAMPLE_AMOUNT) ] )
    df_examples.reset_index(inplace=True, drop=True)
    df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, str(dataset_file).split("_")[0]+"_random_paraphrases.json" ), orient = "index", index = True, indent = 4)   
    
print("Done.")