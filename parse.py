from concurrent.futures import process
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
import json
import random

pd.set_option("display.max_colwidth", None)
filter_strings = ["\n", "", " "]    # the strings we want to filter out

# For debugging or selecting specific datasets to parse:
# DATASETS = ["MSCOCO"]

def parse_datasets():
    '''
    Parses all datasets specified in DATASETS to "output/true_data.json" in a unified format.
    :return: the statistics about the data during the parsing
    '''

    df = pd.DataFrame(columns= [DATASET, ORIGIN, PAIR_ID, ID1, ID2, TEXT1, TEXT2, PARAPHRASE, PARAPHRASE_TYPE] )
    filtered_str = "FILTERING STATS \n\n"

    for dataset in DATASETS:
        path_to_dataset = os.path.join(DATASETS_FOLDER, dataset)
        print("Processing dataset: " + str(path_to_dataset))

        counter = 0
        filtered_amount = 0

        df_tmp = pd.DataFrame(columns= [DATASET, ORIGIN, PAIR_ID, ID1, ID2, TEXT1, TEXT2, PARAPHRASE, PARAPHRASE_TYPE] )

        if dataset == "MPC":
            mpcbert_og_path = os.path.join(path_to_dataset, "og")      #read og data
            mpcbert_mg_path = os.path.join(path_to_dataset, "longformer-large-4096_parallel_mlm_prob_0.15", "mg")      #read og data
            processed_texts = 0
            og_lines = []
            mg_lines = []
            # First read in all dataset lines
            for j, origin_folder in enumerate(os.listdir(mpcbert_og_path)):
                print("Reading " + str(origin_folder))
                for i, file in enumerate(tqdm(os.listdir(os.path.join(mpcbert_og_path, origin_folder)))):
                    with open(os.path.join(mpcbert_og_path, origin_folder, file), encoding="utf8", mode = "r") as f1:
                        with open(os.path.join(mpcbert_mg_path, origin_folder, str(file.replace("ORIG", "SPUN"))), encoding="utf8", mode = "r") as f2:
                            og_line = f1.readlines()
                            og_line = [line.rstrip() for line in og_line]
                            og_line = [l for l in og_line if l != ""][0]
                            mg_line = f2.readlines()
                            mg_line = [line.rstrip() for line in mg_line]
                            mg_line = [l for l in mg_line if l != ""][0]

                            og_lines.append(og_line)
                            mg_lines.append(mg_line)

            # shuffle both lists with the same order keeping them aligned (random sampling)
            print("Shuffle dataset entries to produce random sampling...")
            temp = list(zip(og_lines, mg_lines))
            random.shuffle(temp)
            og_lines, mg_lines = zip(*temp)
            og_lines, mg_lines = list(og_lines), list(mg_lines)

            for i, og_line in tqdm(enumerate(og_lines), total=len(og_lines)):
                mg_line = mg_lines[i]
                if og_line not in filter_strings and mg_line not in filter_strings and og_line != mg_line:
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
                    if df_tmp.shape[0] >= MAX_DATASET_INPUT:    # stop (do not process all)
                        print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                        break
                else:
                    filtered_amount = filtered_amount + 1
            df_tmp.reset_index(drop=True, inplace=True)
            filtered_str = filtered_str + str(dataset) + ": " + str(len(og_lines)) + "\n"
            filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
            filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"

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
                processed_texts = 0
                for i, elem in enumerate(tqdm(root)):
                    paraphrase_types_list = [type_dict[TYPE_ID] for type_dict in paraphrase_types[elem[0].text][PARAPHRASE_TYPE] ]
                    
                    if elem[3].text not in filter_strings and elem[4].text not in filter_strings and elem[3].text != elem[4].text:
                        df_tmp.loc[processed_texts] = np.array([dataset, "newswire", shortuuid.uuid()[:8], elem[1].text, elem[2].text, elem[3].text, elem[4].text, bool(int(elem[8].text)), paraphrase_types_list], dtype=object)
                        processed_texts = processed_texts + 1
                        if processed_texts >= MAX_DATASET_INPUT:
                            print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                            break
                    else:
                        filtered_amount = filtered_amount + 1
                df_tmp.reset_index(drop=True, inplace=True)
                filtered_str = filtered_str + str(dataset) + ": " + str(len(root)) + "\n"
                filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
                filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"

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

                    # shuffle both lists with the same order keeping them aligned (random sampling)
                    print("Shuffle dataset entries to produce random sampling...")
                    temp = list(zip(og_lines, mg_lines))
                    random.shuffle(temp)
                    og_lines, mg_lines = zip(*temp)
                    og_lines, mg_lines = list(og_lines), list(mg_lines)

                    print("Read dataset entries...")
                    for i, og_line in enumerate(tqdm(og_lines)):
                        if og_line.split("\t")[2] not in filter_strings and mg_lines[i].split("\t")[2] not in filter_strings and og_line.split("\t")[2] != mg_lines[i].split("\t")[2]:
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
                            if df_tmp.shape[0] >= MAX_DATASET_INPUT:
                                print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                                break
                        else:
                            filtered_amount = filtered_amount + 1
                    df_tmp.reset_index(drop=True, inplace=True)
                    filtered_str = filtered_str + str(dataset) + ": " + str(len(og_lines)) + "\n"
                    filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
                    filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"
        
        elif dataset == "QQP":
            qqp_path = os.path.join(path_to_dataset, "questions.csv")  
            quora_df = pd.read_csv(qqp_path)
            quora_df = quora_df.sample(frac=1)  # shuffle for random sampling
            for i, row in tqdm(quora_df.iterrows(), total=quora_df.shape[0]):
                if row["question1"] not in filter_strings and row["question2"] not in filter_strings and row["question1"] != row["question2"]:
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
                    if df_tmp.shape[0] >= MAX_DATASET_INPUT:
                        print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                        break
                else:
                    filtered_amount = filtered_amount + 1
            df_tmp.reset_index(drop=True, inplace=True)
            filtered_str = filtered_str + str(dataset) + ": " + str(quora_df.shape[0]) + "\n"
            filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
            filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"

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
                    random.shuffle(lines)   # shuffle for random sampling

                    for i, line in enumerate(tqdm(lines)):
                        if line != "\n" and line.split("\t")[0] not in filter_strings and line.split("\t")[1] not in filter_strings and line.split("\t")[0] != line.split("\t")[1] and int(line.split("\t")[2][1]) != 3:   # if amazon workers could not decide, skip (3/6)
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
                            if df_tmp.shape[0] >= MAX_DATASET_INPUT:
                                print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                                break
                        else:
                            filtered_amount = filtered_amount + 1
                    df_tmp.reset_index(drop=True, inplace=True)
                    filtered_str = filtered_str + str(dataset) + ": " + str(len(lines)) + "\n"
                    filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
                    filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"
        
        elif dataset == "ParaNMT":
            nmt_path = os.path.join(path_to_dataset)      #read train data
            with open(os.path.join(nmt_path, "para-nmt-50m.txt"), encoding="utf8", mode = "r") as f:   # read file line-by-line as it is very big
                lines = f.readlines()
                random.shuffle(lines)   # shuffle for random sampling
                for i, line in tqdm(enumerate(lines), total=len(lines)):
                    l = line.rstrip().split("\t")
                    if l[0] != l[1] and l[0] not in filter_strings and l[1] not in filter_strings:    # only keep paragram-phrase score middleground to remove near identical and noisy data
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
                        if df_tmp.shape[0] >= MAX_DATASET_INPUT:
                            print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                            break
                    else:
                        filtered_amount = filtered_amount + 1
                df_tmp.reset_index(drop=True, inplace=True)
                filtered_str = filtered_str + str(dataset) + ": " + str(len(lines)) + "\n"
                filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
                filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"
        
        elif dataset == "APT":
            apt_path = os.path.join(path_to_dataset)
            processed_texts = 0
            print("Reading MSRP split...")
            with open(os.path.join(apt_path, "apt5-m"), encoding="utf8", mode = "r") as f1: 
                lines = f1.readlines()
                lines = [line.rstrip() for line in lines]
                lines = [l for l in lines if l != ""][1:]
                random.shuffle(lines)   # shuffle for random sampling
                for i, line in enumerate(tqdm(lines)):
                    if line.split("\t")[0] not in filter_strings and line.split("\t")[1] not in filter_strings and line.split("\t")[0] != line.split("\t")[1]:
                        is_paraphrase = bool(int(line.split("\t")[2]))
                        df_tmp.loc[processed_texts] = np.array([
                            dataset, 
                            "msrp", 
                            shortuuid.uuid()[:8], 
                            shortuuid.uuid()[:8], 
                            shortuuid.uuid()[:8], 
                            line.split("\t")[0], 
                            line.split("\t")[1], 
                            is_paraphrase,
                            [0]
                        ], dtype=object)
                        processed_texts = processed_texts + 1
                        if df_tmp.shape[0] >= MAX_DATASET_INPUT / 2:  
                            print("\nReached the max. amount (half-way): " + str(MAX_DATASET_INPUT))
                            break
                    else:
                        filtered_amount = filtered_amount + 1
                df_tmp.reset_index(drop=True, inplace=True)
                filtered_str = filtered_str + str(dataset) + " (msrp split): " + str(len(lines)) + "\n"
                filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
                filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"
            amount_msrp_split = int(df_tmp.shape[0])
            
            print("Reading Twitter PPDB split...")
            with open(os.path.join(apt_path, "apt5-tw"), encoding="utf8", mode = "r") as f1:  
                lines = f1.readlines()
                lines = [line.rstrip() for line in lines]
                lines = [l for l in lines if l != ""][1:]
                random.shuffle(lines)   # shuffle for random sampling
                for i, line in enumerate(tqdm(lines)):
                    if line.split("\t")[0] not in filter_strings and line.split("\t")[1] not in filter_strings and line.split("\t")[0] != line.split("\t")[1]:
                        is_paraphrase = bool(int(line.split("\t")[2]))
                        df_tmp.loc[processed_texts] = np.array([
                            dataset, 
                            "twitterppdb", 
                            shortuuid.uuid()[:8], 
                            shortuuid.uuid()[:8], 
                            shortuuid.uuid()[:8], 
                            line.split("\t")[0], 
                            line.split("\t")[1], 
                            is_paraphrase,
                            [0]
                        ], dtype=object)
                        processed_texts = processed_texts + 1
                        if df_tmp.shape[0] >= MAX_DATASET_INPUT: 
                            print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                            break
                    else:
                        filtered_amount = filtered_amount + 1
                df_tmp.reset_index(drop=True, inplace=True)
                filtered_str = filtered_str + str(dataset) + " (ppdb split): " + str(len(lines)) + "\n"
                filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0] - amount_msrp_split) + "\n"
                filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"

        elif dataset == "APH":
            aph_path = os.path.join(path_to_dataset)
            with open(os.path.join(aph_path, "ap-h-test"), encoding="utf8", mode = "r") as f1: 
                with open(os.path.join(aph_path, "ap-h-train"), encoding="utf8", mode = "r") as f2: 
                    test_lines = f1.readlines()
                    test_lines = [line.rstrip() for line in test_lines]
                    test_lines = [l for l in test_lines if l != ""][1:]
                    train_lines = f2.readlines()
                    train_lines = [line.rstrip() for line in train_lines]
                    train_lines = [l for l in train_lines if l != ""][1:]
                    lines = test_lines + train_lines
                    random.shuffle(lines)   # shuffle for random sampling

                    for i, line in enumerate(tqdm(lines)):
                        if line.split("\t")[0] not in filter_strings and line.split("\t")[1] not in filter_strings and line.split("\t")[0] != line.split("\t")[1]:
                            is_paraphrase = bool(int(line.split("\t")[2]))
                            df_tmp.loc[i] = np.array([
                                dataset, 
                                "msrp,ppnmt", 
                                shortuuid.uuid()[:8], 
                                shortuuid.uuid()[:8], 
                                shortuuid.uuid()[:8], 
                                line.split("\t")[0], 
                                line.split("\t")[1], 
                                is_paraphrase,
                                [0]
                            ], dtype=object)
                            if df_tmp.shape[0] >= MAX_DATASET_INPUT: 
                                print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                                break
                        else:
                            filtered_amount = filtered_amount + 1
                    df_tmp.reset_index(drop=True, inplace=True)
                    filtered_str = filtered_str + str(dataset) + ": " + str(len(lines)) + "\n"
                    filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
                    filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"
        
        elif dataset == "PAWSWiki":
            paws_path = os.path.join(path_to_dataset)
            with open(os.path.join(paws_path, "dev.tsv"), encoding="utf8", mode = "r") as f1: 
                with open(os.path.join(paws_path, "test.tsv"), encoding="utf8", mode = "r") as f2: 
                    with open(os.path.join(paws_path, "train.tsv"), encoding="utf8", mode = "r") as f3: 
                        # read train, test and dev files
                        dev_lines = f1.readlines()
                        dev_lines = [line.rstrip() for line in dev_lines]
                        dev_lines = ["dev"+l for l in dev_lines if l != ""][1:]
                        test_lines = f2.readlines()
                        test_lines = [line.rstrip() for line in test_lines]
                        test_lines = ["test"+l for l in test_lines if l != ""][1:]
                        train_lines = f3.readlines()
                        train_lines = [line.rstrip() for line in train_lines]
                        train_lines = ["train"+l for l in train_lines if l != ""][1:]
                        lines = test_lines + train_lines + dev_lines
                        random.shuffle(lines)   # shuffle for random sampling

                        for i, line in enumerate(tqdm(lines)):
                            if line.split("\t")[0] not in filter_strings and line.split("\t")[1] not in filter_strings and line.split("\t")[0] != line.split("\t")[1]:
                                is_paraphrase = bool(int(line.split("\t")[3]))
                                df_tmp.loc[i] = np.array([
                                    dataset, 
                                    "wikipedia", 
                                    line.split("\t")[0]+"_"+shortuuid.uuid()[:8], 
                                    shortuuid.uuid()[:8], 
                                    shortuuid.uuid()[:8], 
                                    line.split("\t")[1], 
                                    line.split("\t")[2], 
                                    is_paraphrase,
                                    [0]
                                ], dtype=object)
                                if df_tmp.shape[0] >= MAX_DATASET_INPUT: 
                                    print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                                    break
                            else:
                                filtered_amount = filtered_amount + 1
            df_tmp.reset_index(drop=True, inplace=True)
            filtered_str = filtered_str + str(dataset) + ": " + str(len(lines)) + "\n"
            filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
            filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"

        elif dataset == "ParaSCI":
            parasci_paths = [os.path.join(path_to_dataset, "ParaSCI-ACL"), os.path.join(path_to_dataset, "ParaSCI-arXiv")]
            processed_texts = 0

            # --> ACL split
            all_og_lines = []
            all_hg_lines = []
            print("Processing ACL split...")
            with open(os.path.join(parasci_paths[0], "test", "test.src"), encoding="utf8", mode = "r") as f1: 
                with open(os.path.join(parasci_paths[0], "test", "test.tgt"), encoding="utf8", mode = "r") as f2: 
                    og_lines = f1.readlines()
                    og_lines = [line.rstrip() for line in og_lines]
                    hg_lines = f2.readlines()
                    hg_lines = [line.rstrip() for line in hg_lines]
                    all_og_lines = all_og_lines + og_lines
                    all_hg_lines = all_hg_lines + hg_lines
            with open(os.path.join(parasci_paths[0], "train", "train.src"), encoding="utf8", mode = "r") as f1: 
                with open(os.path.join(parasci_paths[0], "train", "train.tgt"), encoding="utf8", mode = "r") as f2: 
                    og_lines = f1.readlines()
                    og_lines = [line.rstrip() for line in og_lines]
                    hg_lines = f2.readlines()
                    hg_lines = [line.rstrip() for line in hg_lines]
                    all_og_lines = all_og_lines + og_lines
                    all_hg_lines = all_hg_lines + hg_lines
            with open(os.path.join(parasci_paths[0], "val", "val.src"), encoding="utf8", mode = "r") as f1: 
                with open(os.path.join(parasci_paths[0], "val", "val.tgt"), encoding="utf8", mode = "r") as f2: 
                    og_lines = f1.readlines()
                    og_lines = [line.rstrip() for line in og_lines]
                    hg_lines = f2.readlines()
                    hg_lines = [line.rstrip() for line in hg_lines]
                    all_og_lines = all_og_lines + og_lines
                    all_hg_lines = all_hg_lines + hg_lines
            
            # shuffle both lists with the same order keeping them aligned (random sampling)
            print("Shuffle dataset entries to produce random sampling...")
            temp = list(zip(all_og_lines, all_hg_lines))
            random.shuffle(temp)
            all_og_lines, all_hg_lines = zip(*temp)
            all_og_lines, all_hg_lines = list(all_og_lines), list(all_hg_lines)
            
            print("Reading data...")
            for i, og_line in tqdm(enumerate(all_og_lines), total=len(all_og_lines)):
                hg_line = all_hg_lines[i]
                if og_line not in filter_strings and hg_line not in filter_strings and og_line != hg_line:
                    df_tmp.loc[processed_texts] = np.array([
                        dataset, 
                        "ACL", 
                        shortuuid.uuid()[:8], 
                        shortuuid.uuid()[:8], 
                        shortuuid.uuid()[:8], 
                        og_line, 
                        hg_line, 
                        True, 
                        [0]
                        ], dtype=object)
                    processed_texts = processed_texts + 1
                    if df_tmp.shape[0] >= MAX_DATASET_INPUT / 2:    # stop (do not process all)
                        break
                else:
                    filtered_amount = filtered_amount + 1
            df_tmp.reset_index(drop=True, inplace=True)
            filtered_str = filtered_str + str(dataset) + " (ACL split): " + str(len(all_og_lines)) + "\n"
            filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
            filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"
            amount_acl_split = int(df_tmp.shape[0])

            # --> arXiv split
            all_og_lines = []
            all_hg_lines = []
            print("Processing arXiv split...")
            with open(os.path.join(parasci_paths[1], "test", "test.src"), encoding="utf8", mode = "r") as f1: 
                with open(os.path.join(parasci_paths[1], "test", "test.tgt"), encoding="utf8", mode = "r") as f2: 
                    og_lines = f1.readlines()
                    og_lines = [line.rstrip() for line in og_lines]
                    hg_lines = f2.readlines()
                    hg_lines = [line.rstrip() for line in hg_lines]
                    all_og_lines = all_og_lines + og_lines
                    all_hg_lines = all_hg_lines + hg_lines
            with open(os.path.join(parasci_paths[1], "train", "train.src"), encoding="utf8", mode = "r") as f1: 
                with open(os.path.join(parasci_paths[1], "train", "train.tgt"), encoding="utf8", mode = "r") as f2: 
                    og_lines = f1.readlines()
                    og_lines = [line.rstrip() for line in og_lines]
                    hg_lines = f2.readlines()
                    hg_lines = [line.rstrip() for line in hg_lines]
                    all_og_lines = all_og_lines + og_lines
                    all_hg_lines = all_hg_lines + hg_lines
            with open(os.path.join(parasci_paths[1], "val", "val.src"), encoding="utf8", mode = "r") as f1: 
                with open(os.path.join(parasci_paths[1], "val", "val.tgt"), encoding="utf8", mode = "r") as f2: 
                    og_lines = f1.readlines()
                    og_lines = [line.rstrip() for line in og_lines]
                    hg_lines = f2.readlines()
                    hg_lines = [line.rstrip() for line in hg_lines]
                    all_og_lines = all_og_lines + og_lines
                    all_hg_lines = all_hg_lines + hg_lines
            
            # shuffle both lists with the same order keeping them aligned (random sampling)
            print("Shuffle dataset entries to produce random sampling...")
            temp = list(zip(all_og_lines, all_hg_lines))
            random.shuffle(temp)
            all_og_lines, all_hg_lines = zip(*temp)
            all_og_lines, all_hg_lines = list(all_og_lines), list(all_hg_lines)
            
            print("Reading data...")
            for i, og_line in tqdm(enumerate(all_og_lines), total=len(all_og_lines)):
                hg_line = all_hg_lines[i]
                if og_line not in filter_strings and hg_line not in filter_strings and og_line != hg_line:
                    df_tmp.loc[processed_texts] = np.array([
                        dataset, 
                        "arXiv", 
                        shortuuid.uuid()[:8], 
                        shortuuid.uuid()[:8], 
                        shortuuid.uuid()[:8], 
                        og_line, 
                        hg_line, 
                        True, 
                        [0]
                        ], dtype=object)
                    processed_texts = processed_texts + 1
                    if df_tmp.shape[0] >= MAX_DATASET_INPUT:    # stop (do not process all)
                        print("\nReached the max. amount (depends on how many ACL entries got included before): " + str(MAX_DATASET_INPUT - amount_acl_split))
                        break
                else:
                    filtered_amount = filtered_amount + 1
            df_tmp.reset_index(drop=True, inplace=True)
            filtered_str = filtered_str + str(dataset) + " (arXiv split): " + str(len(all_og_lines)) + "\n"
            filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
            filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"
        
        elif dataset == "MSCOCO":
            # after https://ojs.aaai.org/index.php/AAAI/article/view/11956
            # every image has been annotated by multiple annotators. Whe combine these image captions to multiple versions of paraphrased pairs.
            coco_train_path = os.path.join(path_to_dataset, "annotations", "captions_train2017.json")
            coco_val_path = os.path.join(path_to_dataset, "annotations", "captions_val2017.json")

            with open(coco_train_path, "r") as f:
                train_dict = json.load(f)

            captions_df = pd.read_json(json.dumps(train_dict["annotations"]))
            img_ids = captions_df["image_id"].unique()
            random.shuffle(img_ids)     # sample randomly
            print("Found unique images: " + str(len(img_ids)))

            print("Reading data...")
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                this_img_df = captions_df[captions_df["image_id"] == img_id].reset_index(drop=True)
                random_caption_id_1 = random.randint(0,this_img_df.shape[0]-1)    # sample two annotators randomly from all five annotators
                random_caption_id_2 = random.randint(0,this_img_df.shape[0]-1)    
                while random_caption_id_2 == random_caption_id_1:
                    random_caption_id_2 = random.randint(0,this_img_df.shape[0]-1)
                og_line = this_img_df.iloc[random_caption_id_1]["caption"]
                hg_line = this_img_df.iloc[random_caption_id_2]["caption"]

                if og_line not in filter_strings and hg_line not in filter_strings and og_line != hg_line:
                    df_tmp.loc[i] = np.array([
                        dataset, 
                        "imgcaptions", 
                        str(img_id)+"_"+shortuuid.uuid()[:8], 
                        shortuuid.uuid()[:8], 
                        shortuuid.uuid()[:8], 
                        og_line, 
                        hg_line, 
                        True, 
                        [0]
                        ], dtype=object)
                    if df_tmp.shape[0] >= MAX_DATASET_INPUT:    # stop (do not process all)
                        print("\nReached the max. amount: " + str(MAX_DATASET_INPUT))
                        break
                else:
                    filtered_amount = filtered_amount + 1
            df_tmp.reset_index(drop=True, inplace=True)
            filtered_str = filtered_str + str(dataset) + " (unique image ids): " + str(len(img_ids)) + "\n"
            filtered_str = filtered_str + "after filtering: " + str(df_tmp.shape[0]) + "\n"
            filtered_str = filtered_str + "filtered pairs: " + str(filtered_amount) + "\n\n"



        df = pd.concat([df, df_tmp], ignore_index = True)   # concat the lastly processed dataset to the combined dataset

        print("Current Stats: \n\n")
        print(filtered_str)

    #Output data to json format
    df.to_json(os.path.join(OUT_DIR, "small_data.json"), orient = "index", index = True, indent = 4)
    print("Done.")

    return filtered_str


if __name__ == "__main__":
    filtered_str = parse_datasets()
    with open(os.path.join(OUT_DIR, "stats_parsing_filtering.txt"), "w") as text_file:
        text_file.write(filtered_str)