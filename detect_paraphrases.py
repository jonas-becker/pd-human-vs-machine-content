import os
import pandas as pd
from tqdm import tqdm
from re import sub
import numpy as np
from thefuzz import fuzz
import shortuuid
from setup import *
import xml.etree.ElementTree as ET
from sklearn.utils import shuffle
import re
import sys
from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex, SoftCosineSimilarity, Similarity

DATASETS_FOLDER = "datasets"    #the folder that contains the dataset directories to read in
FORMATTED_DATA_FILENAME = "true_data.json"  #the name of the file that contains the data to read in
OUT_DIR = "output"      #the directory to output the formatted json in

FUZZY = "fuzzy_based_result"
SEMANTIC = "semantic_based_result"

STOPWORDS = ['the', 'and', 'are', 'a']


def preprocess(doc):
    # Tokenize and clean data
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in STOPWORDS]

def check_semantic(corpus, string_2, similarity_matrix, tfidf, dictionary):
    # returns the semantic similarity based on the corpus
    query = preprocess(string_2)
    query_tf = tfidf[dictionary.doc2bow(query)]
    index = SoftCosineSimilarity(tfidf[[dictionary.doc2bow(document) for document in corpus]], similarity_matrix)
    return index[query_tf]

def semantic_sim(df):
    corpus = [ preprocess(document) for document in list(df["text_1"]) ]
    # use a pre trained model: https://huggingface.co/fse/glove-wiki-gigaword-50 , https://nlp.stanford.edu/pubs/glove.pdf
    glove = api.load("glove-wiki-gigaword-50")
    similarity_index = WordEmbeddingSimilarityIndex(glove)
    # Build the term dictionary and the tfidf model
    dictionary = Dictionary(corpus)
    tfidf = TfidfModel(dictionary=dictionary)
    # Create the term similarity matrix.    
    print("Creating the similarity matrix...")
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)     #takes a long time
    print("Processing texts...")
    semantic_results = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        sim = check_semantic(corpus, row["text_2"], similarity_matrix, tfidf, dictionary)
        try:
            semantic_results.append(sim[i])
        except Exception as e:
            #print("result is " + str(sim) + ". Appending the only result value: " + str(float(sim.item())))
            #print(e)
            semantic_results.append(float(sim.item()))
            continue
    return semantic_results

def fuzzy_sim(df):
    #Check for paraphrase with fuzzy based
    fuzzy_results = []
    print("Checking for paraphrases with the fuzzy-based method. Dataframe rows to process: " + str(len(df)))
    for i, row in tqdm(df.iterrows()):
        fuzzy_results.append(float(fuzz.ratio(row["text_1"], row["text_2"])/100))
    return fuzzy_results

#################################################################################


for embedded_file in os.listdir(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER)):
    print(f"Processing {embedded_file}...")
    df = pd.read_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, embedded_file), orient = "index")
    df = df[(df[TEXT1] != "") & (df[TEXT2] != "")].reset_index(drop=True)
    print(f"{df.shape[0]} pairs found in the embedded dataset file.")
    #df = df.truncate(after=500)  #cut part of dataframe for testing
    dataset = df.iloc[0][DATASET]

    # Modify datasets to be balanced (paraphrase & non-paraphrase)
    if len(df[df[PARAPHRASE] == False]) < 10:    # if there is no noteworthy amount of original pairs, add them from other datasets (e.g. machine-datasets)
        # get paraphrase types for all pair IDs (read from different files)
        paraphrase_types = {}
        for filler_dataset in FILLER_DATASETS:
            df_tmp = pd.read_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, filler_dataset+"_embedded.json"), orient= "index")
            print(df_tmp.shape[0])
            df_tmp = df_tmp[df_tmp[PARAPHRASE] == False]
            print(f"Adding {df_tmp.shape[0]} original pairs from {filler_dataset} to {dataset} for balancing.")
            df = pd.concat([df, df_tmp], ignore_index = True)   #concat the lastly processed dataset to the combined dataset

    # Truncate "to much" data for balancing
    df = shuffle(df).reset_index(drop=True)
    df = df.groupby(PARAPHRASE)
    df = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True)))
    df = shuffle(df).reset_index(drop=True)
    print("Balanced dataset with the following paraphrased-statistics:")
    print(df[PARAPHRASE].value_counts())

    # Calculate the similarities with each method
    df[FUZZY] = fuzzy_sim(df)
    df[SEMANTIC] = semantic_sim(df)

    #Output data to json format
    df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, embedded_file.split("_")[0]+"_result.json"), orient = "index", index = True, indent = 4)

