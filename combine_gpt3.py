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

df = pd.read_csv(os.path.join(OUT_DIR, "gpt3-remaining-embeddings.csv"))

with zipfile.ZipFile(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, 'embeddings-gpt-3.zip'), 'a') as archive:
    
    print("archive length before: " + str(len(archive.namelist())))
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if i <= 3:
            continue
        filename1 = f"{row[PAIR_ID]}_text_1.txt"
        filename2 = f"{row[PAIR_ID]}_text_2.txt"
        str1 = row["text_1_gpt3_embedding"].replace("[","").replace("]","").replace(", ", "\n")
        str2 = row["text_2_gpt3_embedding"].replace("[","").replace("]","").replace(", ", "\n")
        with open("tmp1.txt", "w") as f1:
            f1.write(str1)
        with open("tmp2.txt", "w") as f2:
            f2.write(str2)

        archive.write( "tmp1.txt", "embeddings-gpt-3/"+os.path.basename(filename1))
        archive.write( "tmp2.txt", "embeddings-gpt-3/"+os.path.basename(filename2))


print("Done.")
print("archive length now: " + str(len(archive.namelist())))
print("added pairs: " + str(len(df)))