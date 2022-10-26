import os
import pandas as pd
from tqdm import tqdm
from re import sub
import numpy as np
from thefuzz import fuzz
from setup import *
from strsimpy.ngram import NGram
import numpy
from sklearn.metrics import roc_auc_score
import io
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import PrecisionRecallDisplay
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import zipfile
from transformers import AutoTokenizer
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# from classification.text_mining.Evaluation import Classification
# from classification.text_mining.CommandLine import CLClassification
# from classification.text_mining.Evaluation import Classification
# from classification.text_mining.GridSearch import GridSearch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import math

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def preprocess_function_text1(df):
    return bert_tokenizer(df[TEXT1], truncation=True)


def preprocess_function_text2(df):
    return bert_tokenizer(df[TEXT2], truncation=True)


def append_result(df, test_result_df, test_pair_ids, test_pred_result):
    test_result_df = pd.concat(test_result_df)

    return test_result_df

def semantic_sim_bert(df, pair_ids_train, pair_ids_test, text1_train, text1_test, text2_train, text2_test, y_train,
                      y_test):
    print("Calculating semantic similarity with BERT.")
    print("Train examples: " + str(len(pair_ids_train)))
    print("Test examples: " + str(len(pair_ids_test)))
    model = SentenceTransformer('all-distilroberta-v1')
    print("Created model.")

    print(sum(map(lambda x: x is True, y_test)))
    print(sum(map(lambda x: x is False, y_test)))

    print("Creating embeddings for text_1s (test).")
    X1_test = model.encode(text1_test)
    print("Creating embeddings for text_2s (test).")
    X2_test = model.encode(text2_test)
    X_test = np.column_stack((X1_test, X2_test))

    print("Creating embeddings for text_1s (training).")
    X1_train = model.encode(text1_train)
    print("Creating embeddings for text_2s (training).")
    X2_train = model.encode(text2_train)
    X_train = np.column_stack((X1_train, X2_train))

    '''
    with zipfile.ZipFile(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, 'embeddings-BERT.zip'), 'a') as archive:
        print("Exporting embeddings (text 1)...")
        for i, text1_embedding in tqdm(enumerate(X1_test)):
            with open("tmp.txt", "w") as f1:
                f1.write(np.array2string(numpy.array(text1_embedding), separator='\n'))
                archive.write( "tmp.txt", os.path.basename(df.iloc[i][PAIR_ID]+"_text_1.txt"))
        print("Exporting embeddings (text 2)...")
        for i, text2_embedding in tqdm(enumerate(X2_test)):
            with open("tmp.txt", "w") as f2:
                f2.write(np.array2string(numpy.array(text2_embedding), separator='\n'))
                archive.write("tmp.txt", os.path.basename(df.iloc[i][PAIR_ID]+"_text_2.txt"))
    '''

    print("Processing texts...")
    # take care, maybe train data has to be processed in batches
    class_weight = compute_class_weight(
        class_weight='balanced', classes=[False, True], y=y_train
    )
    print("Weighting classes: " + str(class_weight))

    # initialize the model and assign weights to each class
    # clf = SVC(class_weight={False: class_weight[0], True: class_weight[1]}, verbose=True)
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=1000, class_weight={False: class_weight[0], True: class_weight[1]},
                                      verbose=True, loss="modified_huber", alpha=1.0)
                        )
    # train the model
    print("Training the SVM and calculating decision function scores for the test split...")

    clf.fit(X_train, y_train)
    # use the model to predict the testing instances
    print("Testing the SVM...")
    y_pred = clf.predict(X_test)
    # generate the classification report
    print(classification_report(y_test, y_pred))

    y_score = clf.decision_function(X_test)

    prediction_result = [sigmoid(y_item) for y_item in y_score.tolist()]  # normalize with sigmoid

    print(prediction_result)

    # Save results to dataframe
    # for i, pair_id in enumerate(pair_ids_test):
    #    df.loc[df[PAIR_ID] == pair_id, SPLIT] = "test"
    #    df.loc[df[PAIR_ID] == pair_id, SEM_BERT] = bool(prediction_result[i])
    return prediction_result


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
                archive.write("tmp.txt", os.path.basename(df.iloc[i][PAIR_ID] + "_text_1.txt"))
        print("Exporting embeddings (text 2)...")
        for i, text2_embedding in tqdm(enumerate(text2_embeddings)):
            with open("tmp.txt", "w") as f2:
                f2.write(np.array2string(numpy.array(text2_embedding), separator='\n'))
                archive.write("tmp.txt", os.path.basename(df.iloc[i][PAIR_ID] + "_text_2.txt"))

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
            if "embeddings-gpt-3/" + str(row[PAIR_ID]) + "_text_1.txt" not in archive.namelist():
                sim = None
            else:
                with io.TextIOWrapper(archive.open("embeddings-gpt-3/" + str(row[PAIR_ID]) + "_text_1.txt")) as f1:
                    text1_embedding = f1.read()  # [2:]
                    t1_embed = []
                    for e in list(text1_embedding.split("\n"))[:-1]:
                        t1_embed.append(float(e))
                with io.TextIOWrapper(archive.open("embeddings-gpt-3/" + str(row[PAIR_ID]) + "_text_2.txt")) as f2:
                    text2_embedding = f2.read()  # [2:]
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
        sim = ngram.distance(corpus1[i], corpus2[i])  # calculate sim between text1 and text2 pairwise
        results.append(sim)
    return results


def fuzzy_sim(df):
    # Check for paraphrase with fuzzy based
    fuzzy_results = []
    print("Checking for paraphrases with the fuzzy-based method.")
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        fuzzy_results.append(float(fuzz.ratio(row[TEXT1], row[TEXT2]) / 100))
    return fuzzy_results


def tfidf_cosine_sim(df):
    print("Calculating TF-IDF cosine similarities.")
    corpus1 = list(df[TEXT1])
    corpus2 = list(df[TEXT2])

    print("Processing texts...")
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(corpus1 + corpus2)  # combine text1 and text2 to one corpus

    results = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        sim = cosine_similarity(tf_idf_matrix[i],
                                tf_idf_matrix[len(corpus1) + i])  # calculate sim between text1 and text2 pairwise
        results.append(sim[0][0])
    return results


def get_splits(df):
    splits = ["train", "test", "dev", "val", None]
    for dataset in DATASETS:
        print("Dataset: " + str(dataset))
        for s_type in splits:
            amount = df[(df[DATASET] == dataset) & (df[SPLIT] == s_type)].shape[0]
            print(str(s_type) + ": " + str(amount))
        print("---")


#################################################################################

if not os.path.isdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    os.makedirs(os.path.join(OUT_DIR, DETECTION_FOLDER))

stats_str = "STATISTICS OF DETECTION SCRIPT\n\n"

print("Reading " + FORMATTED_DATA_FILENAME + " ...")
df = pd.read_json(os.path.join(OUT_DIR, FORMATTED_DATA_FILENAME), orient="index")
df = df.sort_values(by=[DATASET])  # sort to have datasets processed sequentially

pred_result_df = pd.DataFrame()

#df = df.reset_index(drop=True)
#df = df.truncate(before=0, after=20000)     # for testing
# df = df[(df[DATASET] == "MPC") | (df[DATASET] == "ETPC")]   # for testing

# get_splits(df)

print(f"{df.shape[0]} pairs found in the file.")
# df = df.truncate(after=200)  #cut part of dataframe for testing

# Calculate the similarities with each method
datasets_in_df = df[DATASET].unique().tolist()
total_pred_result = []
print("Found the following datasets in the data: " + str(datasets_in_df))

test_data = [[], [], [], []]
train_data = [[], [], [], []]

for dataset in datasets_in_df:
    print("Classify " + str(dataset) + "...")

    df_dataset = df[df[DATASET] == dataset]
    df_dataset[SUPPLEMENT_FROM] = None

    # For datasets with only positive pairs, supplement the data with random pairs from other datasets
    if False not in df_dataset[PARAPHRASE].unique():
        supplement_df = df[~(df[DATASET] == dataset)].sample(frac=1).reset_index(drop=True)
        supplement_df = supplement_df[~supplement_df[PARAPHRASE]].head(df_dataset.shape[0])
        supplement_df[SUPPLEMENT_FROM] = supplement_df[DATASET]
        supplement_df[DATASET] = dataset
        df_dataset = pd.concat([df_dataset, supplement_df])
        print("Supplemented dataset. It does now contain " + str(df_dataset.shape[0]) + " pairs.")
    df_dataset.reset_index(drop=True, inplace=True)
    print(df_dataset[PARAPHRASE].value_counts())

    print("Managing features...")
    feature_pairs = list(df_dataset[PAIR_ID])
    feature_pairs = [[x] + [df_dataset[TEXT1].iloc[i]] + [df_dataset[TEXT2].iloc[i]] + [df_dataset[PARAPHRASE].iloc[i]]
                     for i, x in tqdm(enumerate(feature_pairs), total=len(feature_pairs))]

    print(len(feature_pairs))

    X = np.array(feature_pairs)
    y = np.array(df_dataset[PARAPHRASE])

    print("Creating splits...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    sss.get_n_splits()

    print("Assembling data...")
    for train_index, test_index in sss.split(X, y):
        test_data[0] = test_data[0] + X[test_index][:, 0].tolist()  # pair ids
        test_data[1] = test_data[1] + X[test_index][:, 1].tolist()  # text 1
        test_data[2] = test_data[2] + X[test_index][:, 2].tolist()  # text 2
        test_data[3] = test_data[3] + [True if x == "True" else False for x in X[test_index][:, 3].tolist()]  # labels
        train_data[0] = train_data[0] + X[train_index][:, 0].tolist()  # pair ids
        train_data[1] = train_data[1] + X[train_index][:, 1].tolist()  # text 1
        train_data[2] = train_data[2] + X[train_index][:, 2].tolist()  # text 2
        train_data[3] = train_data[3] + [True if x == "True" else False for x in
                                         X[train_index][:, 3].tolist()]  # labels

        print(sum(map(lambda x: x == True, train_data[3])))
        print(sum(map(lambda x: x == False, train_data[3])))
        print("---")
        print(sum(map(lambda x: x == True, test_data[3])))
        print(sum(map(lambda x: x == False, test_data[3])))

        # add only to df if pair is not included already, this does not include results yet
        pred_result_df = pd.concat([pred_result_df, df_dataset[df_dataset.index.isin(test_index)]])

# use classifiers
print("Finished assembling data. Continue with classification...")
pred_result_df[SEM_BERT] = semantic_sim_bert(df, train_data[0], test_data[0], train_data[1], test_data[1],
                                             train_data[2], test_data[2], train_data[3], test_data[3])

# df[TFIDF_COSINE] = tfidf_cosine_sim(df)
# df[NGRAM3] = ngram_sim(df, 3)
# df[FUZZY] = fuzzy_sim(df)
# df[SEM_GPT3] = semantic_sim_gpt3(df)
# df[SEM_T5] = semantic_sim_t5(df)

# Output data to json format
# df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_result.json"), orient = "index", index = True, indent = 4)
print(len(pred_result_df))
pred_result_df = pred_result_df.reset_index(drop=True)
print(len(pred_result_df))
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)

print("Done.")
