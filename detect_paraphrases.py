import os
import pandas as pd
from tqdm import tqdm
from re import sub
import numpy as np
from thefuzz import fuzz
from setup import *
import xml.etree.ElementTree as ET
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import sys
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex, SoftCosineSimilarity, Similarity

def preprocess(doc):
    # Tokenize and clean data
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in STOPWORDS]

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def check_semantic(corpus, string_2, similarity_matrix, tfidf, dictionary):
    # returns the semantic similarity based on the corpus
    query = preprocess(string_2)
    query_tf = tfidf[dictionary.doc2bow(query)]
    index = SoftCosineSimilarity(tfidf[[dictionary.doc2bow(document) for document in corpus]], similarity_matrix)
    return index[query_tf]

def semantic_sim_glove(df):
    print("Calculating semantic similarity with GLoVe.")
    corpus = [ preprocess(document) for document in list(df[TEXT1]) ]
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
        sim = check_semantic(corpus, row[TEXT2], similarity_matrix, tfidf, dictionary)
        try:
            semantic_results.append(sim[i])
        except Exception as e:
            #print("result is " + str(sim) + ". Appending the only result value: " + str(float(sim.item())))
            #print(e)
            semantic_results.append(float(sim.item()))
            continue
    return semantic_results

def semantic_sim_bert(df):
    print("Calculating semantic similarity with BERT.")
    corpus1 = list(df[TEXT1])
    corpus2 = list(df[TEXT2])
    # use bert to embed
    model = SentenceTransformer('bert-base-uncased')
    print("Encoding text_1's...")
    text1_embeddings = model.encode(corpus1)
    print("Encoding text_2's...")
    text2_embeddings = model.encode(corpus2)

    print("Processing texts...")
    semantic_results = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        sim = cosine_similarity([text1_embeddings[i]], [text2_embeddings[i]])[0][0]
        try:
            semantic_results.append(sim)
        except Exception as e:
            semantic_results.append(float(sim.item()))
            continue
    return semantic_results

def semantic_sim_t5(df):
    print("Calculating semantic similarity with T5.")
    corpus1 = list(df[TEXT1])
    corpus2 = list(df[TEXT2])
    # use bert to embed
    model = SentenceTransformer('sentence-t5-base')
    print("Encoding text_1's...")
    text1_embeddings = model.encode(corpus1)
    print("Encoding text_2's...")
    text2_embeddings = model.encode(corpus2)

    print("Processing texts...")
    semantic_results = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        sim = cosine_similarity([text1_embeddings[i]], [text2_embeddings[i]])[0][0]
        try:
            semantic_results.append(sim)
        except Exception as e:
            semantic_results.append(float(sim.item()))
            continue
    return semantic_results

def semantic_sim_t5(df):
    print("Calculating semantic similarity with T5.")
    corpus1 = list(df[TEXT1])
    corpus2 = list(df[TEXT2])
    # use bert to embed
    model = SentenceTransformer('sentence-t5-base')
    print("Encoding text_1's...")
    text1_embeddings = model.encode(corpus1)
    print("Encoding text_2's...")
    text2_embeddings = model.encode(corpus2)

    print("Processing texts...")
    semantic_results = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        sim = cosine_similarity([text1_embeddings[i]], [text2_embeddings[i]])[0][0]
        try:
            semantic_results.append(sim)
        except Exception as e:
            semantic_results.append(float(sim.item()))
            continue
    return semantic_results

def semantic_sim_gpt3(df):
    # TODO: Write this function
    print("Calculating semantic similarity with GPT-3.")
    # Add a function that calculates the semantic similarity of each text pair within the dataframe 
    # (similar to the above semantic similarity functions)
    # use GPT-3

    # Should return a list semantic_results (float type: 0 being least similar and 1 being identical pair)
    # Should use cosine similarity 
    semantic_results = []

    return semantic_results


'''
def ngram_sim(df):
    print("Calculating similarity with N-Grams and their TF-IDF cosine similarity.")
    corpus1 = list(df[TEXT1])
    corpus2 = list(df[TEXT2])

    print("Processing texts...")
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(corpus1+corpus2)   # combine text1 and text2 to one corpus
    
    results = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        sim = cosine_similarity(tf_idf_matrix[i], tf_idf_matrix[len(corpus1)+i])    #calculate sim between text1 and text2 pairwise
        results.append(sim[0][0])
    return results
'''

def fuzzy_sim(df):
    #Check for paraphrase with fuzzy based
    fuzzy_results = []
    print("Checking for paraphrases with the fuzzy-based method. Dataframe rows to process: " + str(len(df)))
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        fuzzy_results.append(float(fuzz.ratio(row[TEXT1], row[TEXT2])/100))
    return fuzzy_results

def tfidf_cosine_sim(df):
    print("Calculating TF-IDF cosine similarities.")
    corpus1 = list(df[TEXT1])
    corpus2 = list(df[TEXT2])

    print("Processing texts...")
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(corpus1+corpus2)   # combine text1 and text2 to one corpus
    
    results = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        sim = cosine_similarity(tf_idf_matrix[i], tf_idf_matrix[len(corpus1)+i])    #calculate sim between text1 and text2 pairwise
        results.append(sim[0][0])
    return results


#################################################################################

for embedded_file in os.listdir(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER)):
    print(f"Processing {embedded_file}...")
    df = pd.read_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, embedded_file), orient = "index")
    df = df[(df[TEXT1] != "") & (df[TEXT2] != "")].reset_index(drop=True)
    print(f"{df.shape[0]} pairs found in the embedded dataset file.")
    #df = df.truncate(after=500)  #cut part of dataframe for testing
    dataset = df.iloc[0][DATASET]

    # Modify datasets to be balanced (paraphrase & non-paraphrase)
    '''
    if len(df[df[PARAPHRASE] == False]) < 10:    # if there is no noteworthy amount of original pairs, add them from other datasets (e.g. machine-datasets)
        # get paraphrase types for all pair IDs (read from different files)
        paraphrase_types = {}
        for filler_dataset in FILLER_DATASETS:
            df_tmp = pd.read_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, filler_dataset+"_embedded.json"), orient= "index")
            df_tmp = df_tmp[df_tmp[PARAPHRASE] == False]
            print(f"Adding {df_tmp.shape[0]} original pairs from {filler_dataset} to {dataset} for balancing.")
            df = pd.concat([df, df_tmp], ignore_index = True)   #concat the lastly processed dataset to the combined dataset
    '''

    # Truncate "to much" data for balancing
    if len(df[df[PARAPHRASE] == False]) < 10:   # only truncate datasets that are not paraphrase-pairs only
        df = shuffle(df).reset_index(drop=True)
        df = df.groupby(PARAPHRASE)
        df = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True)))
    df = shuffle(df).reset_index(drop=True)
    print("Balanced dataset with the following paraphrased-statistics:")
    print(df[PARAPHRASE].value_counts())
 
    # Calculate the similarities with each method
    df[TFIDF_COSINE] = tfidf_cosine_sim(df)
    # df[NGRAM] = ngram_sim(df)  # not working reliably for strings of different length, so leave out for now
    df[FUZZY] = fuzzy_sim(df)
    df[SEM_BERT] = semantic_sim_bert(df)
    df[SEM_T5] = semantic_sim_t5(df)
    df[SEM_GPT3] = semantic_sim_gpt3(df)
    #df[SEM_GLOVE] = semantic_sim_glove(df)

    #Output data to json format
    df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, embedded_file.split("_")[0]+"_result.json"), orient = "index", index = True, indent = 4)

print("Done.")
