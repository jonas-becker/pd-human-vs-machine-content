import os
import pandas as pd
from tqdm import tqdm
from re import sub
import numpy as np
from thefuzz import fuzz
from setup import *
from strsimpy.ngram import NGram
import numpy
import io
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import zipfile
from transformers import AutoTokenizer

bert_tokenizer = AutoTokenizer.from_pretrained("destilbert-base-uncased")

def preprocess_function_text1(df):
    return bert_tokenizer(df[TEXT1], truncation=True)
def preprocess_function_text2(df):
    return bert_tokenizer(df[TEXT2], truncation=True)
def semantic_sim_bert_new(df):
    print("Calculating semantic similarity with BERT.")
    tokenized_text1s = df.map(preprocess_function_text1, batched=True)
    tokenized_text2s = df.map(preprocess_function_text2, batched=True)

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

    with zipfile.ZipFile(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, 'embeddings-BERT.zip'), 'a') as archive:
        print("Exporting embeddings (text 1)...")
        for i, text1_embedding in tqdm(enumerate(text1_embeddings)):
            with open("tmp.txt", "w") as f1:
                f1.write(np.array2string(numpy.array(text1_embedding), separator='\n'))
                archive.write( "tmp.txt", os.path.basename(df.iloc[i][PAIR_ID]+"_text_1.txt"))
        print("Exporting embeddings (text 2)...")
        for i, text2_embedding in tqdm(enumerate(text2_embeddings)):
            with open("tmp.txt", "w") as f2:
                f2.write(np.array2string(numpy.array(text2_embedding), separator='\n'))
                archive.write("tmp.txt", os.path.basename(df.iloc[i][PAIR_ID]+"_text_2.txt"))

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

    with zipfile.ZipFile(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, 'embeddings-T5.zip'), 'a') as archive:
        print("Exporting embeddings (text 1)...")
        for i, text1_embedding in tqdm(enumerate(text1_embeddings)):
            with open("tmp.txt", "w") as f1:
                f1.write(np.array2string(numpy.array(text1_embedding), separator='\n'))
                archive.write( "tmp.txt", os.path.basename(df.iloc[i][PAIR_ID]+"_text_1.txt"))
        print("Exporting embeddings (text 2)...")
        for i, text2_embedding in tqdm(enumerate(text2_embeddings)):
            with open("tmp.txt", "w") as f2:
                f2.write(np.array2string(numpy.array(text2_embedding), separator='\n'))
                archive.write( "tmp.txt", os.path.basename(df.iloc[i][PAIR_ID]+"_text_2.txt"))

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
    print("Calculating semantic similarity with GPT-3.")
    semantic_results = []
    with zipfile.ZipFile(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, 'embeddings-gpt-3.zip'), 'r') as archive:
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            if "embeddings-gpt-3/"+str(row[PAIR_ID])+"_text_1.txt" not in archive.namelist():
                sim = None
            else:
                with io.TextIOWrapper(archive.open("embeddings-gpt-3/"+str(row[PAIR_ID])+"_text_1.txt")) as f1:
                    text1_embedding = f1.read()#[2:]
                    t1_embed = []
                    for e in list(text1_embedding.split("\n"))[:-1]:
                        t1_embed.append(float(e))
                with io.TextIOWrapper(archive.open("embeddings-gpt-3/"+str(row[PAIR_ID])+"_text_2.txt")) as f2:
                    text2_embedding = f2.read()#[2:]
                    t2_embed = []
                    for e in list(text2_embedding.split("\n"))[:-1]:
                        t2_embed.append(float(e))
                sim = cosine_similarity([t1_embed], [t2_embed])[0][0]
            semantic_results.append(sim)

    return semantic_results

def ngram_sim(df, n):
    # done after http://webdocs.cs.ualberta.ca/~kondrak/papers/spire05.pdf
    print(f"Calculating similarity with {n}-Grams.")
    corpus1 = list(df[TEXT1])
    corpus2 = list(df[TEXT2])

    print("Processing texts...")
    results = []
    ngram = NGram(n)
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        sim = ngram.distance(corpus1[i], corpus2[i])    #calculate sim between text1 and text2 pairwise
        results.append(sim)
    return results

def fuzzy_sim(df):
    #Check for paraphrase with fuzzy based
    fuzzy_results = []
    print("Checking for paraphrases with the fuzzy-based method.")
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

if not os.path.isdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    os.makedirs(os.path.join(OUT_DIR, DETECTION_FOLDER))

stats_str = "STATISTICS OF DETECTION SCRIPT\n\n"

for embedded_file in os.listdir(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER)):

    if ".zip" in embedded_file:
        continue

    print(f"Processing {embedded_file}...")
    df = pd.read_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, embedded_file), orient = "index")
    df = df[(df[TEXT1] != "") & (df[TEXT2] != "")].reset_index(drop=True)
    print(f"{df.shape[0]} pairs found in the embedded dataset file.")
    #df = df.truncate(after=200)  #cut part of dataframe for testing
    dataset = df.iloc[0][DATASET]

    # Truncate "to much" data for balancing
    if len(df[df[PARAPHRASE] == False]) > 10:   # only truncate datasets that are not paraphrase-pairs only
        df = shuffle(df).reset_index(drop=True)
        df = df.groupby(PARAPHRASE)
        df = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True)))
    df = shuffle(df).reset_index(drop=True)

    print("Balanced dataset with the following paraphrased-statistics:")
    print(df[PARAPHRASE].value_counts())
    stats_str = stats_str + "Dataset: " + str(embedded_file) + "\nBalanced dataset with the following paraphrased-statistics: \n" + str(df[PARAPHRASE].value_counts()) + "\n\n"
    with open(os.path.join(OUT_DIR, "stats_detection_script.txt"), "w") as text_file:
        text_file.write(stats_str)

    # Calculate the similarities with each method
    df[TFIDF_COSINE] = tfidf_cosine_sim(df)
    df[NGRAM3] = ngram_sim(df, 3)
    df[FUZZY] = fuzzy_sim(df)
    df[SEM_GPT3] = semantic_sim_gpt3(df)
    df[SEM_BERT] = semantic_sim_bert(df)
    df[SEM_T5] = semantic_sim_t5(df)

    #Output data to json format
    df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, embedded_file.split("_")[0]+"_result.json"), orient = "index", index = True, indent = 4)

print("Done.")
