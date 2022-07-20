import os
import pandas as pd
from setup import *
from sklearn.utils import shuffle

def get_embed_examples():
    for dataset_file in os.listdir(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER)):
        df_examples = pd.DataFrame()

        print("Processing " + dataset_file)
        df = pd.read_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, dataset_file), orient = "index")
        # filter out 1.0 distances (exactly the same texts)
        df = df[df[COSINE_DISTANCE] != 1.0]

        # (paraphrase)
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(COSINE_DISTANCE, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(COSINE_DISTANCE, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )

        #Output data to json format
        df_examples.reset_index(inplace=True, drop=True)
        df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "embeddings", str(dataset_file).split("_")[0]+"_top_paraphrases.json" ), orient = "index", index = True, indent = 4)
        df_examples = pd.DataFrame()

        # (non-paraphrase)
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(COSINE_DISTANCE, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(COSINE_DISTANCE, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )

        #Output data to json format
        df_examples.reset_index(inplace=True, drop=True)
        df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "embeddings", str(dataset_file).split("_")[0]+"_top_originals.json" ), orient = "index", index = True, indent = 4)
        df_examples = pd.DataFrame()

        # random pairs (paraphrase)

        df = shuffle(df)    #shuffle the df
        # randoms original pairs
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].head(EXAMPLE_AMOUNT) ] )
        df_examples.reset_index(inplace=True, drop=True)
        df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "embeddings", str(dataset_file).split("_")[0]+"_random_originals.json" ), orient = "index", index = True, indent = 4)    
        df_examples = pd.DataFrame()

        # randoms paraphrased pairs
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].head(EXAMPLE_AMOUNT) ] )
        df_examples.reset_index(inplace=True, drop=True)
        df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "embeddings", str(dataset_file).split("_")[0]+"_random_paraphrases.json" ), orient = "index", index = True, indent = 4)   
    
def get_detection_examples():
    for dataset_file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
        print("Processing " + dataset_file)
        df_examples = pd.DataFrame()

        df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, dataset_file), orient = "index")
        # filter out 1.0 distances (exactly the same texts)
        df = df[df[COSINE_DISTANCE] != 1.0]
        
        for detection_method in DETECTION_METHODS:
            print("Method: " + detection_method)

            # (paraphrase)
            df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(detection_method, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )
            df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(detection_method, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )

            #Output data to json format
            df_examples.reset_index(inplace=True, drop=True)
            df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection", str(dataset_file).split("_")[0]+"_"+detection_method+"_paraphrases.json" ), orient = "index", index = True, indent = 4)
            df_examples = pd.DataFrame()

            # (non-paraphrase)
            df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(detection_method, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )
            df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(detection_method, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )

            #Output data to json format
            df_examples.reset_index(inplace=True, drop=True)
            df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection", str(dataset_file).split("_")[0]+"_"+detection_method+"_originals.json" ), orient = "index", index = True, indent = 4)
            df_examples = pd.DataFrame()


    print("Getting examples from all datasets combined...")
    df = pd.DataFrame()
    for dataset_file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
        df = pd.concat([df, pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, dataset_file), orient = "index")])
    
    for detection_method in DETECTION_METHODS:
        print("Method: " + detection_method)
        df_examples = pd.DataFrame()

        # (paraphrase)
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(detection_method, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].sort_values(detection_method, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )

        #Output data to json format
        df_examples.reset_index(inplace=True, drop=True)
        df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection", "total_"+detection_method+"_paraphrases.json" ), orient = "index", index = True, indent = 4)
        df_examples = pd.DataFrame()

        # (non-paraphrase)
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(detection_method, axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )
        df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].sort_values(detection_method, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').head(EXAMPLE_AMOUNT) ] )

        #Output data to json format
        df_examples.reset_index(inplace=True, drop=True)
        df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection", "total_"+detection_method+"_originals.json" ), orient = "index", index = True, indent = 4)
        df_examples = pd.DataFrame()
    
    # random pairs (paraphrase)
    df = shuffle(df)    #shuffle the df
    # randoms original pairs
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == False].head(EXAMPLE_AMOUNT) ] )
    df_examples.reset_index(inplace=True, drop=True)
    df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection", "total_random_originals.json" ), orient = "index", index = True, indent = 4)    
    df_examples = pd.DataFrame()

    # randoms paraphrased pairs
    df_examples = pd.concat( [ df_examples, df[df[PARAPHRASE] == True].head(EXAMPLE_AMOUNT) ] )
    df_examples.reset_index(inplace=True, drop=True)
    df_examples.to_json(os.path.join(OUT_DIR, EXAMPLES_FOLDER, "detection", "total_random_paraphrases.json" ), orient = "index", index = True, indent = 4) 

print("Welcome!")
print("1 - Embedding Examples")
print("2 - Detection Examples")

x = int(input("Please enter a number to define what you want to get examples from: "))

if x == 1:
    print("Getting Embedding Examples...")
    get_embed_examples()
elif x == 2:
    print("Getting Detection Examples...")
    get_detection_examples()
else:
    print("Your input seems wrong. Please enter a valid number.")

print("Done.")