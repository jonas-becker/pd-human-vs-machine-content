import os
import pandas as pd
from tqdm import tqdm
import shortuuid
import xml.etree.ElementTree as ET
import re
import sys
import string
from setup import *
import numpy as np

pd.set_option("display.max_colwidth", None)

# For Debugging:
DATASETS = ["ParaNMT"]

#Bring datasets to the same format (standardized)

df = pd.DataFrame(columns= [DATASET, ORIGIN, PAIR_ID, ID1, ID2, TEXT1, TEXT2, PARAPHRASE, PARAPHRASE_TYPE] )

for dataset in DATASETS:
    path_to_dataset = os.path.join(DATASETS_FOLDER, dataset)
    print("Processing dataset: " + str(path_to_dataset))

    counter = 0

    df_tmp = pd.DataFrame(columns= [DATASET, ORIGIN, PAIR_ID, ID1, ID2, TEXT1, TEXT2, PARAPHRASE, PARAPHRASE_TYPE] )

    if dataset == "MPC":
        dmop_path = os.path.join(path_to_dataset, "wikipedia_documents_train", "machined")      #read train data
        for file in tqdm(os.listdir(os.path.join(dmop_path, "og"))):
            with open(os.path.join(dmop_path, "og", file), encoding="utf8", mode = "r") as f1:
                with open(os.path.join(dmop_path, "mg", str(file.split("-")[0])+"-SPUN.txt"), encoding="utf8", mode = "r") as f2:
                    og_lines = f1.readlines()
                    og_lines = [line.rstrip() for line in og_lines]
                    og_lines = [l for l in og_lines if l != ""]
                    mg_lines = f2.readlines()
                    mg_lines = [line.rstrip() for line in mg_lines]
                    mg_lines = [l for l in mg_lines if l != ""]

                    if len(og_lines) != len(mg_lines):
                        print("ERROR")

                    for i, og_line in enumerate(og_lines):
                        #counter = counter+1
                        #if counter > 30:
                        #    break

                        if og_line != "\n":
                            df_tmp.loc[i] = np.array([dataset, "wikipedia", shortuuid.uuid()[:8], shortuuid.uuid()[:8], shortuuid.uuid()[:8], og_line, mg_lines[i], True, [0]], dtype=object)
        
        df = pd.concat([df, df_tmp], ignore_index = True)
        df_tmp = pd.DataFrame(columns= [DATASET, ORIGIN, PAIR_ID, ID1, ID2, TEXT1, TEXT2, PARAPHRASE, PARAPHRASE_TYPE] )

        dmop_path = os.path.join(path_to_dataset, "wikipedia_documents_test", "machined")
        for file in tqdm(os.listdir(os.path.join(dmop_path, "og"))):        #read test data (combine as there is no ML process involved)
            with open(os.path.join(dmop_path, "og", file), encoding="utf8", mode = "r") as f1:
                with open(os.path.join(dmop_path, "mg", str(file.split("-")[0])+"-SPUN.txt"), encoding="utf8", mode = "r") as f2:
                    og_lines = f1.readlines()
                    og_lines = [line.rstrip() for line in og_lines]
                    og_lines = [l for l in og_lines if l != ""]
                    mg_lines = f2.readlines()
                    mg_lines = [line.rstrip() for line in mg_lines]
                    mg_lines = [l for l in mg_lines if l != ""]

                    for i, og_line in enumerate(og_lines):
                        #counter = counter+1
                        #if counter > 30:
                        #    break
                        if og_line != "\n":
                            df_tmp.loc[i] = np.array([dataset, "wikipedia", shortuuid.uuid()[:8], shortuuid.uuid()[:8], shortuuid.uuid()[:8], og_line, mg_lines[i], True, [0]], dtype=object)
    
    elif dataset == "MPCBert":
        mpcbert_og_path = os.path.join(path_to_dataset, "og")      #read og data
        mpcbert_mg_path = os.path.join(path_to_dataset, "bert-large-cased_parallel_mlm_prob_0.3", "mg")      #read og data
        processed_texts = 0
        for j, origin_folder in enumerate(os.listdir(mpcbert_og_path)):
            print("Reading " + str(origin_folder))
            for i, file in enumerate(tqdm(os.listdir(os.path.join(mpcbert_og_path, origin_folder)))):
                #counter = counter+1
                #if i > 30:
                #    break
                with open(os.path.join(mpcbert_og_path, origin_folder, file), encoding="utf8", mode = "r") as f1:
                    with open(os.path.join(mpcbert_mg_path, origin_folder, str(file.replace("ORIG", "SPUN"))), encoding="utf8", mode = "r") as f2:
                        og_line = f1.readlines()
                        og_line = [line.rstrip() for line in og_line]
                        og_line = [l for l in og_line if l != ""][0]
                        mg_line = f2.readlines()
                        mg_line = [line.rstrip() for line in mg_line]
                        mg_line = [l for l in mg_line if l != ""][0]

                        if og_line != "\n":
                            df_tmp.loc[processed_texts] = np.array([
                                dataset, 
                                str(origin_folder).split("_")[0], 
                                shortuuid.uuid()[:8], 
                                shortuuid.uuid()[:8], 
                                shortuuid.uuid()[:8], 
                                og_line, 
                                mg_line, 
                                True, 
                                [0]
                                ], dtype=object)
                processed_texts = processed_texts + 1

    elif dataset == "ETPC":
        # get paraphrase types for all pair IDs (read from different files)
        paraphrase_types = {}
        with open(os.path.join(path_to_dataset, "textual_paraphrases.xml"), encoding='utf-8', mode = "r") as file:
            tree = ET.parse(file)
            root = tree.getroot()
            for i, elem in enumerate(root):
                if elem[0].text in paraphrase_types.keys():
                    paraphrase_types[elem[0].text] = { PARAPHRASE_TYPE:  paraphrase_types[elem[0].text][PARAPHRASE_TYPE] + 
                    [ { TYPE_ID: int(elem[1].text), SENSE_PRESERVING: bool(elem[2].text == "yes"), TEXT1_SCOPE: elem[4].text, TEXT2_SCOPE: elem[5].text } ] }
                else:
                    paraphrase_types[elem[0].text] = { PARAPHRASE_TYPE: [ { TYPE_ID: int(elem[1].text), SENSE_PRESERVING: bool(elem[2].text == "yes"), TEXT1_SCOPE: elem[4].text, TEXT2_SCOPE: elem[5].text } ] }
        with open(os.path.join(path_to_dataset, "textual_np_neg.xml"), encoding='utf-8', mode = "r") as file:
            tree = ET.parse(file)
            root = tree.getroot()
            for i, elem in enumerate(root):
                if elem[0].text in paraphrase_types.keys():
                    paraphrase_types[elem[0].text] = { PARAPHRASE_TYPE:  paraphrase_types[elem[0].text][PARAPHRASE_TYPE] + 
                    [ { TYPE_ID: int(elem[1].text), SENSE_PRESERVING: bool(elem[2].text == "yes"), TEXT1_SCOPE: elem[4].text, TEXT2_SCOPE: elem[5].text } ] }
                else:
                    paraphrase_types[elem[0].text] = { PARAPHRASE_TYPE: [ { TYPE_ID: int(elem[1].text), SENSE_PRESERVING: bool(elem[2].text == "yes"), TEXT1_SCOPE: elem[4].text, TEXT2_SCOPE: elem[5].text } ] }
        with open(os.path.join(path_to_dataset, "textual_np_pos.xml"), encoding='utf-8', mode = "r") as file:
            tree = ET.parse(file)
            root = tree.getroot()
            for i, elem in enumerate(root):
                if elem[0].text in paraphrase_types.keys():
                    paraphrase_types[elem[0].text] = { PARAPHRASE_TYPE:  paraphrase_types[elem[0].text][PARAPHRASE_TYPE] + 
                    [ { TYPE_ID: int(elem[1].text), SENSE_PRESERVING: bool(elem[2].text == "yes"), TEXT1_SCOPE: elem[4].text, TEXT2_SCOPE: elem[5].text } ] }
                else:
                    paraphrase_types[elem[0].text] = { PARAPHRASE_TYPE: [ { TYPE_ID: int(elem[1].text), SENSE_PRESERVING: bool(elem[2].text == "yes"), TEXT1_SCOPE: elem[4].text, TEXT2_SCOPE: elem[5].text } ] }
        with open(os.path.join(path_to_dataset, "negation.xml"), encoding='utf-8', mode = "r") as file:
            tree = ET.parse(file)
            root = tree.getroot()
            for i, elem in enumerate(root):
                if elem[0].text in paraphrase_types.keys():
                    paraphrase_types[elem[0].text] = { PARAPHRASE_TYPE:  paraphrase_types[elem[0].text][PARAPHRASE_TYPE] + 
                    [ { TYPE_ID: int(elem[1].text), SENSE_PRESERVING: bool(elem[2].text == "yes"), TEXT1_SCOPE: elem[4].text, TEXT2_SCOPE: elem[5].text } ] }
                else:
                    paraphrase_types[elem[0].text] = { PARAPHRASE_TYPE: [ { TYPE_ID: int(elem[1].text), SENSE_PRESERVING: bool(elem[2].text == "yes"), TEXT1_SCOPE: elem[4].text, TEXT2_SCOPE: elem[5].text } ] }

        # get text pairs and assign the type data
        with open(os.path.join(path_to_dataset, "text_pairs.xml"), encoding='utf-8', mode = "r") as file:
            tree = ET.parse(file)
            root = tree.getroot()
            for i, elem in enumerate(tqdm(root)):
                #counter = counter+1
                #if counter > 30:
                #    break
                paraphrase_types_list = [type_dict[TYPE_ID] for type_dict in paraphrase_types[elem[0].text][PARAPHRASE_TYPE] ]
                df_tmp.loc[i] = np.array([dataset, "newswire", shortuuid.uuid()[:8], elem[1].text, elem[2].text, elem[3].text, elem[4].text, bool(30 in paraphrase_types_list), paraphrase_types_list], dtype=object)

    elif dataset == "SAv2":
        asv2_path = os.path.join(path_to_dataset) 
        with open(os.path.join(asv2_path, "normal.aligned"), encoding="utf8", mode = "r") as f1:
            with open(os.path.join(asv2_path, "simple.aligned"), encoding="utf8", mode = "r") as f2:
                og_lines = f1.readlines()
                og_lines = [line.rstrip() for line in og_lines]
                og_lines = [l for l in og_lines if l != ""]
                mg_lines = f2.readlines()
                mg_lines = [line.rstrip() for line in mg_lines]
                mg_lines = [l for l in mg_lines if l != ""]

                for i, og_line in enumerate(tqdm(og_lines)):
                    #counter = counter+1
                    #if counter > 30:
                    #    break
                    if og_line != "\n":
                        df_tmp.loc[i] = np.array([
                            dataset, 
                            "wikipedia",
                            shortuuid.uuid()[:8],
                            og_line.split("\t")[0].translate(str.maketrans('', '', string.punctuation+" ")) + "_" + shortuuid.uuid()[:8], 
                            mg_lines[i].split("\t")[0].translate(str.maketrans('', '', string.punctuation+" ")) + "_" + shortuuid.uuid()[:8], 
                            og_line.split("\t")[2], 
                            mg_lines[i].split("\t")[2], 
                            True,
                            [16]    # simplification dataset ( => only ellipsis)
                        ], dtype=object)
    
    elif dataset == "QQP":
        qqp_path = os.path.join(path_to_dataset, "questions.csv")  
        quora_df = pd.read_csv(qqp_path)
        for i, row in tqdm(quora_df.iterrows(), total=quora_df.shape[0]):
            #counter = counter+1
            #   if counter > 30:
            #       break
            df_tmp.loc[i] = np.array([
                dataset, 
                "quora",
                str(row["id"]) + "_" + shortuuid.uuid()[:8],
                str(row["qid1"]) + "_" + shortuuid.uuid()[:8], 
                str(row["qid2"]) + "_" + shortuuid.uuid()[:8], 
                row["question1"], 
                row["question2"], 
                bool(row["is_duplicate"]),
                [0]     # unknown type
            ], dtype=object)

    elif dataset == "TURL":
        turl_path = os.path.join(path_to_dataset)    
        with open(os.path.join(turl_path, "Twitter_URL_Corpus_test.txt"), encoding="utf8", mode = "r") as f1:
            with open(os.path.join(turl_path, "Twitter_URL_Corpus_train.txt"), encoding="utf8", mode = "r") as f2:
                test_lines = f1.readlines()
                test_lines = [line.rstrip() for line in test_lines]
                test_lines = [l for l in test_lines if l != ""]
                train_lines = f2.readlines()
                train_lines = [line.rstrip() for line in train_lines]
                train_lines = [l for l in train_lines if l != ""]
                lines = test_lines + train_lines

                for i, line in enumerate(tqdm(lines)):
                    #counter = counter+1
                    #if counter > 30:
                    #    break
                    if line != "\n" and int(line.split("\t")[2][1] != 3):   # if amazon workers could not decide, skip
                        # based on the datasets paper, we value a phrase as paraphrase when >=4 out of 6 amazon workers marked it a such
                        is_paraphrase = int(line.split("\t")[2][1]) >= 4
                        df_tmp.loc[i] = np.array([
                            dataset, 
                            "twitter news", 
                            shortuuid.uuid()[:8], 
                            shortuuid.uuid()[:8], 
                            shortuuid.uuid()[:8], 
                            line.split("\t")[0], 
                            line.split("\t")[1], 
                            is_paraphrase,
                            [0]
                        ], dtype=object)
    
    elif dataset == "ParaNMT":
        nmt_path = os.path.join(path_to_dataset)      #read train data
        print("Reading lines of " + str(os.path.join(nmt_path, "para-nmt-50m.txt")) + " until " + str(MAX_DATASET_INPUT) + " entries with a fitting parapgram-phrase score have been found. Range: " + str(PARAGRAM_PHRASE_SCORE))
        with open(os.path.join(nmt_path, "para-nmt-50m.txt"), encoding="utf8", mode = "r") as f:   # read file line-by-line as it is very big
            for i, line in tqdm(enumerate(f)):
                l = line.rstrip().split("\t")
                if PARAGRAM_PHRASE_SCORE[0] < float(l[2]) < PARAGRAM_PHRASE_SCORE[1]:    # only keep paragram-phrase score middleground to remove near identical and noisy data
                    df_tmp.loc[i] = np.array([
                            dataset, 
                            "czeng", 
                            shortuuid.uuid()[:8], 
                            shortuuid.uuid()[:8], 
                            shortuuid.uuid()[:8], 
                            l[0], 
                            l[1], 
                            True,
                            [0]
                        ], dtype=object)
                if df_tmp.shape[0] >= MAX_DATASET_INPUT:    # stop (do not process al 5 million)
                    break


    df = pd.concat([df, df_tmp], ignore_index = True)   #concat the lastly processed dataset to the combined dataset

#Output data to json format
df.to_json(os.path.join(OUT_DIR, "true_data.json"), orient = "index", index = True, indent = 4)

print("Done.")