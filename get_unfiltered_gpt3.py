import os
from sqlite3 import DatabaseError
import pandas as pd
from tqdm import tqdm
import shortuuid
import xml.etree.ElementTree as ET
import re
import sys
import string
from setup import *
import numpy as np
import zipfile

df = pd.read_json(os.path.join(OUT_DIR, FORMATTED_DATA_FILENAME), orient = "index")

print(df.head())
new_df = pd.DataFrame(columns= [DATASET, ORIGIN, PAIR_ID, ID1, ID2, TEXT1, TEXT2, PARAPHRASE, PARAPHRASE_TYPE] )

already_done = []

with zipfile.ZipFile(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, 'embeddings-gpt-3.zip'), 'r') as archive:

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        
        if "embeddings-gpt-3/"+str(row[PAIR_ID])+"_text_1.txt" not in archive.namelist():
            if len(list(row[PARAPHRASE_TYPE])) > 1:
                new_df = pd.concat([new_df, pd.DataFrame(
                    {
                    DATASET: row[DATASET], 
                    ORIGIN: row[ORIGIN], 
                    PAIR_ID: row[PAIR_ID], 
                    ID1: row[ID1], 
                    ID2: row[ID2], 
                    TEXT1: row[TEXT1], 
                    TEXT2: row[TEXT2], 
                    PARAPHRASE: row[PARAPHRASE], 
                    PARAPHRASE_TYPE: row[PARAPHRASE_TYPE]
                    }
                ) ], ignore_index = True, )
            else:
                new_df = pd.concat([new_df, pd.DataFrame(
                    {
                        DATASET: row[DATASET], 
                        ORIGIN: row[ORIGIN], 
                        PAIR_ID: row[PAIR_ID], 
                        ID1: row[ID1], 
                        ID2: row[ID2], 
                        TEXT1: row[TEXT1], 
                        TEXT2: row[TEXT2], 
                        PARAPHRASE: row[PARAPHRASE], 
                        PARAPHRASE_TYPE: [row[PARAPHRASE_TYPE]]
                    }        
                ) ], ignore_index = True, )
        else:
            if row[DATASET] not in already_done:
                already_done.append(row[DATASET])
                print("Detected dataset already gpt3d: " + str(row[DATASET]))

        if i % 10000 == 0:
            new_df.reset_index(inplace=True, drop=True)
            print(new_df.head())
            print("Iteration: " + str(i))
            print("archive length: " + str(len(archive.namelist())))
            print("total: " + str(len(df)))
            print("now: " + str(new_df.shape[0]))

new_df.reset_index(inplace=True, drop=True)
new_df.to_json(os.path.join(OUT_DIR, "rest_to_process_with_gpt3.json"), orient = "index", index = True, indent = 4)

print("Done.")
print("archive length: " + str(len(archive.namelist())))
print("total: " + str(len(df)))
print("now: " + str(len(new_df)))

print("DATASETS ALREADY DONE: " + str(already_done))