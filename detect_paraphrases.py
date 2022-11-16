import os
import pandas as pd
from sklearn.svm import SVC
from tqdm import tqdm
from re import sub
import numpy as np
from thefuzz import fuzz
from setup import *
from sklearn.model_selection import GridSearchCV
from strsimpy.ngram import NGram
from gensim.models import fasttext
import numpy
from sklearn.metrics import roc_auc_score
from torch import nn
import io
from gensim.scripts.glove2word2vec import glove2word2vec
import matplotlib.pyplot as plt
import pprint
from transformers import BertModel, T5Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display
from sentence_transformers import SentenceTransformer
import zipfile
import fasttext
import tensorflow as tf
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import GridSearchCV
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import torch
import math
from tensorflow.python.ops.numpy_ops import np_config
import gc
gc.collect()

np_config.enable_numpy_behavior()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device " + str(device))
torch.cuda.empty_cache()


def GridSearch_table_plot(grid_clf, param_name, method_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):

    '''Display grid search results

    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          '''

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    scores_df.to_csv(os.path.join(OUT_DIR, DETECTION_FOLDER, "grid_search_"+method_name+".csv"))

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.savefig(os.path.join(OUT_DIR, DETECTION_FOLDER, 'grid_search_' + method_name + '.png'))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def semantic_sim_bert(text1_train, text1_test, text2_train, text2_test, y_train, y_test, gs_params, verb, cv, n_jobs):
    print("Semantic Similarity (BERT) \n------------")
    print("Loading model...")
    model = BertModel.from_pretrained("bert-large-uncased").to(device)
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')

    print("Creating embeddings (test split).")

    '''
    # Code using large amounts of GPU memory:
    texts_test = [list(tuple) for tuple in zip(text1_test, text2_test)]
    t_encoded = tokenizer(texts_test, padding=True, truncation=True, return_tensors="pt").to(device)
    print(t_encoded)
    X_test = model(**t_encoded)
    print(X_test)
    '''

    print("Create embeddings (test split)...")
    X1_test = []
    for t in tqdm(text1_test):
        t_encoded = tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding = model(**t_encoded).last_hidden_state.detach().cpu()
        X1_test.append(torch.mean(embedding, 0)[0].tolist())
    X2_test = []
    for t in tqdm(text2_test):
        t_encoded = tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding = model(**t_encoded).last_hidden_state.detach().cpu()
        X2_test.append(torch.mean(embedding, 0)[0].tolist())
    X_test = np.column_stack((X1_test, X2_test))

    print("Create embeddings (train split)...")
    X1_train = []
    for t in tqdm(text1_train):
        t_encoded = tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding = model(**t_encoded).last_hidden_state.detach().cpu()
        X1_train.append(torch.mean(embedding, 0)[0].tolist())
    X2_train = []
    for t in tqdm(text2_train):
        t_encoded = tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding = model(**t_encoded).last_hidden_state.detach().cpu()
        X2_train.append(torch.mean(embedding, 0)[0].tolist())
    X_train = np.column_stack((X1_train, X2_train))

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    # Grid Search
    gs = GridSearchCV(model_svm, gs_params, cv=cv, verbose=verb, n_jobs=n_jobs)
    gs.fit(X_train, y_train)
    GridSearch_table_plot(gs, "C", "T5", negative=False)

    print("Predicting test split...")
    prediction_result = gs.predict_proba(X_test)

    return [p[1] for p in prediction_result]  # only get probability for one of two classes (true/false)

    '''
    # get output of slice layer from model above
    print("Initializing SVM Model...")
    cls_layer_model = Model(model.input, outputs=model.get_layer('tf.__operators__.getitem').output)

    print("Get embeddings...")
    print(train_input.keys())
    X_train = cls_layer_model.predict({"input_word_ids": np.array(train_input["input_ids"]), "input_mask": np.array(train_input["attention_mask"]), "segment_ids": np.array(train_input["token_type_ids"])})
    print(X_train)
    X_test = cls_layer_model.predict({"input_word_ids": np.array(test_input["input_ids"]), "input_mask": np.array(test_input["attention_mask"]), "segment_ids": np.array(test_input["token_type_ids"])})

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)

    # Grid Search
    gs = GridSearchCV(model_svm, gs_params, cv=3, verbose=50)
    gs.fit(X_train, y_train)
    GridSearch_table_plot(gs, "C",  "BERT", negative=False)

    print("Predicting test split...")
    prediction_result = gs.predict_proba(X_test)

    return [p[1] for p in prediction_result]    # only get probability for one of two classes (true/false)
    '''


def semantic_sim_t5(text1_train, text1_test, text2_train, text2_test, y_train, y_test, gs_params, verb, cv, n_jobs):
    print("Semantic Similarity (T5) \n------------")
    print("Loading model...")

    model = T5Model.from_pretrained("t5-large").to(device)
    tokenizer = AutoTokenizer.from_pretrained('t5-large')

    print("Create embeddings (test split)...")
    X1_test = []
    for t in tqdm(text1_test):
        t_encoded = tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding = model.encoder(input_ids=t_encoded["input_ids"], attention_mask=t_encoded["attention_mask"]).last_hidden_state.detach().cpu()
        X1_test.append(torch.mean(embedding, 0)[0].tolist())    # mean all word embeddings to get 1 sentence embedding
    X2_test = []
    for t in tqdm(text2_test):
        t_encoded = tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding = model.encoder(input_ids=t_encoded["input_ids"], attention_mask=t_encoded["attention_mask"]).last_hidden_state.detach().cpu()
        X2_test.append(torch.mean(embedding, 0)[0].tolist())
    X_test = np.column_stack((X1_test, X2_test))

    print("Create embeddings (train split)...")
    X1_train = []
    for t in tqdm(text1_train):
        t_encoded = tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding = model.encoder(input_ids=t_encoded["input_ids"], attention_mask=t_encoded["attention_mask"]).last_hidden_state.detach().cpu()
        X1_train.append(torch.mean(embedding, 0)[0].tolist())
    X2_train = []
    for t in tqdm(text2_train):
        t_encoded = tokenizer(t, padding=True, truncation=True, return_tensors="pt").to(device)
        embedding = model.encoder(input_ids=t_encoded["input_ids"], attention_mask=t_encoded["attention_mask"]).last_hidden_state.detach().cpu()
        X2_train.append(torch.mean(embedding, 0)[0].tolist())
    X_train = np.column_stack((X1_train, X2_train))

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    # Grid Search
    gs = GridSearchCV(model_svm, gs_params, cv=cv, verbose=verb, n_jobs=n_jobs)
    gs.fit(X_train, y_train)
    GridSearch_table_plot(gs, "C", "T5", negative=False)

    print("Predicting test split...")
    prediction_result = gs.predict_proba(X_test)

    return [p[1] for p in prediction_result]    # only get probability for one of two classes (true/false)


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


def ngram_sim(n, text1_train, text1_test, text2_train, text2_test, y_train, y_test, gs_params, verb, cv, n_jobs):
    # done after http://webdocs.cs.ualberta.ca/~kondrak/papers/spire05.pdf
    print(f"Calculating similarity with {n}-Grams.")

    print("Processing texts...")
    sims_train = []
    sims_test = []
    ngram = NGram(n)
    for i, row in tqdm(enumerate(text1_train), total=len(text1_train)):
        sim = ngram.distance(text1_train[i], text2_train[i])  # calculate sim between text1 and text2 pairwise
        sims_train.append(sim)
    for i, row in tqdm(enumerate(text1_test), total=len(text1_test)):
        sim = ngram.distance(text1_test[i], text2_test[i])  # calculate sim between text1 and text2 pairwise
        sims_test.append(sim)
    sims_train = np.array(sims_train).reshape(-1, 1)
    sims_test = np.array(sims_test).reshape(-1, 1)


    # Grid Search
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    gs = GridSearchCV(model_svm, gs_params, cv=cv, verbose=verb, n_jobs=n_jobs)

    # train the model
    print("Training the SVM...")
    gs.fit(sims_train, y_train)
    print("Output grid search stats...")
    GridSearch_table_plot(gs, "C", "NGram", negative=False)

    # use the model to predict the testing instances
    print("Testing the SVM...")
    prediction_result = gs.predict_proba(sims_test)

    return [p[1] for p in prediction_result]


def fuzzy_sim(text1_train, text1_test, text2_train, text2_test, y_train, y_test, gs_params, verb, cv, n_jobs):
    print(f"Calculating similarity with Fuzzy.")

    print("Processing texts...")
    sims_train = []
    sims_test = []
    for i, row in tqdm(enumerate(text1_train), total=len(text1_train)):
        sim = float(fuzz.ratio(text1_train[i], text2_train[i]) / 100)  # calculate sim between text1 and text2 pairwise
        sims_train.append(sim)
    for i, row in tqdm(enumerate(text1_test), total=len(text1_test)):
        sim = float(fuzz.ratio(text1_test[i], text2_test[i]) / 100)  # calculate sim between text1 and text2 pairwise
        sims_test.append(sim)
    sims_train = np.array(sims_train).reshape(-1, 1)
    sims_test = np.array(sims_test).reshape(-1, 1)

    # Grid Search
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    gs = GridSearchCV(model_svm, gs_params, cv=cv, verbose=verb, n_jobs=n_jobs)

    # train the model
    print("Training the SVM...")
    gs.fit(sims_train, y_train)
    print("Output grid search stats...")
    GridSearch_table_plot(gs, "C", "NGram", negative=False)

    # use the model to predict the testing instances
    print("Testing the SVM...")
    prediction_result = gs.predict_proba(sims_test)

    return [p[1] for p in prediction_result]


def create_embedding_matrix(word_index, embedding_dict, dimension):
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))

    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix


def semantic_sim_glove(text1_train, text1_test, text2_train, text2_test, y_train, y_test, gs_params, verb, cv, n_jobs):
    print("GloVe Similarity \n------------")
    print("Loading model...")

    glove = pd.read_csv(os.path.join(MODELS_FOLDER, 'glove.6B.100d.txt'), sep=" ", quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    print("Tokenize (fit)...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(split=" ")
    tokenizer.fit_on_texts(text1_test + text1_train + text2_train + text2_test)

    print("Tokenize test split..")
    text1_test_token = tokenizer.texts_to_sequences(text1_test)
    text2_test_token = tokenizer.texts_to_sequences(text2_test)
    print("Tokenize train split...")
    text1_train_token = tokenizer.texts_to_sequences(text1_train)
    text2_train_token = tokenizer.texts_to_sequences(text2_train)

    print("Creating embedding matrix...")
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict=glove_embedding, dimension=100)

    vocab_size = embedding_matrix.shape[0]
    vector_size = embedding_matrix.shape[1]

    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)

    embedding_layer.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    embedding_layer.weight.requires_grad = False  # not trainable weights, use pretrained glove

    X1_test = []
    for t in tqdm(text1_test_token):
        embedding = embedding_layer(torch.LongTensor(t))
        X1_test.append(torch.mean(embedding, 0).tolist())
    X2_test = []
    for t in tqdm(text2_test_token):
        embedding = embedding_layer(torch.LongTensor(t))
        X2_test.append(torch.mean(embedding, 0).tolist())
    X_test = np.column_stack((X1_test, X2_test))
    X_test = np.nan_to_num(X_test)  # fill NaN with zeros

    X1_train = []
    for t in tqdm(text1_train_token):
        embedding = embedding_layer(torch.LongTensor(t))
        X1_train.append(torch.mean(embedding, 0).tolist())
    X2_train = []
    for t in tqdm(text2_train_token):
        embedding = embedding_layer(torch.LongTensor(t))
        X2_train.append(torch.mean(embedding, 0).tolist())
    X_train = np.column_stack((X1_train, X2_train))
    X_train = np.nan_to_num(X_train)  # fill NaN with zeros

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    # Grid Search
    gs = GridSearchCV(model_svm, gs_params, cv=cv, verbose=verb, n_jobs=n_jobs)
    gs.fit(X_train, y_train)
    GridSearch_table_plot(gs, "C", "GloVe", negative=False)

    print("Predicting test split...")
    prediction_result = gs.predict_proba(X_test)

    return [p[1] for p in prediction_result]    # only get probability for one of two classes (true/false)

def fasttext_sim(text1_train, text1_test, text2_train, text2_test, y_train, y_test, gs_params, verb, cv, n_jobs):
    print("GloVe Similarity \n------------")
    print("Loading model...")

    #model = fasttext.load_facebook_vectors(os.path.join(MODELS_FOLDER, "model_filename.bin"))

    model = fasttext.load_model(os.path.join(MODELS_FOLDER, 'cc.en.300.bin'))

    X1_test = []
    for t in tqdm(text1_test):
        vector = model.get_sentence_vector(t.replace("\n", " "))
        X1_test.append(vector)
    X2_test = []
    for t in tqdm(text2_test):
        vector = model.get_sentence_vector(t.replace("\n", " "))
        X2_test.append(vector)
    X_test = np.column_stack((X1_test, X2_test))

    X1_train = []
    for t in tqdm(text1_train):
        vector = model.get_sentence_vector(t.replace("\n", " "))
        X1_train.append(vector)
    X2_train = []
    for t in tqdm(text2_train):
        vector = model.get_sentence_vector(t.replace("\n", " "))
        X2_train.append(vector)
    X_train = np.column_stack((X1_train, X2_train))

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    # Grid Search
    gs = GridSearchCV(model_svm, gs_params, cv=cv, verbose=verb, n_jobs=n_jobs)
    gs.fit(X_train, y_train)
    GridSearch_table_plot(gs, "C", "fasttext", negative=False)

    print("Predicting test split...")
    prediction_result = gs.predict_proba(X_test)

    return [p[1] for p in prediction_result]    # only get probability for one of two classes (true/false)


def tfidf_cosine_sim(text1_train, text1_test, text2_train, text2_test, y_train, y_test, gs_params, verb, cv, n_jobs):
    print("Calculating TF-IDF cosine similarities.")

    print("Processing texts...")

    vectorizer = TfidfVectorizer()
    tf_idf_matrix_train = vectorizer.fit_transform(text1_train + text2_train)  # combine text1 and text2 to one corpus
    tf_idf_matrix_test = vectorizer.fit_transform(text1_test + text2_test)  # combine text1 and text2 to one corpus

    sims_train = []
    sims_test = []
    for i, row in tqdm(enumerate(text1_train), total=len(text1_train)):
        sim = cosine_similarity(tf_idf_matrix_train[i], tf_idf_matrix_train[len(text1_train) + i])
        sims_train.append(sim)
    for i, row in tqdm(enumerate(text1_test), total=len(text1_test)):
        sim = cosine_similarity(tf_idf_matrix_test[i], tf_idf_matrix_test[len(text1_test) + i])
        sims_test.append(sim)
    sims_train = np.array(sims_train).reshape(-1, 1)
    sims_test = np.array(sims_test).reshape(-1, 1)

    # Grid Search
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    gs = GridSearchCV(model_svm, gs_params, cv=cv, verbose=verb, n_jobs=n_jobs)

    # train the model
    print("Training the SVM...")
    gs.fit(sims_train, y_train)
    print("Output grid search stats...")
    GridSearch_table_plot(gs, "C", "NGram", negative=False)

    # use the model to predict the testing instances
    print("Testing the SVM...")
    prediction_result = gs.predict_proba(sims_test)

    return [p[1] for p in prediction_result]

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

print(None in df[TEXT2].tolist())
print(None in df[TEXT1].tolist())

pred_result_df = pd.DataFrame()

df = df.reset_index(drop=True)
df = df.truncate(before=0, after=50000)     # for testing
# df = df[(df[DATASET] == "MPC") | (df[DATASET] == "ETPC")]   # for testing

print(f"{df.shape[0]} pairs found in the file.")

# Calculate the similarities with each method
datasets_in_df = df[DATASET].unique().tolist()
total_pred_result = []
print("Found the following datasets in the data: " + str(datasets_in_df))

test_data = [[], [], [], []]
train_data = [[], [], [], []]

for dataset in datasets_in_df:
    print("Processing " + str(dataset) + "...")

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

    print("Managing pair ids...")

    y = np.array(df_dataset[PARAPHRASE])

    # max amount of train split to make data comparable
    # (rest is used as test split)
    if len(df_dataset) > TRAIN_SPLIT_MAX:
        train_split_size = TRAIN_SPLIT_MAX / len(df_dataset)
    else:
        train_split_size = 0.8
    print("Determined train split size for " + str(dataset) + ": " + str(train_split_size))

    print("Creating splits...")
    df_dataset_train, df_dataset_test, X_pairID_train, X_pairID_test, X_text1_train, X_text1_test, X_text2_train, X_text2_test, y_train, y_test = train_test_split(df_dataset, df_dataset[PAIR_ID], df_dataset[TEXT1], df_dataset[TEXT2], y, train_size=train_split_size, random_state=0, stratify=y)

    print("Assembling data...")
    test_data[0] = test_data[0] + X_pairID_test.tolist()  # pair ids
    test_data[1] = test_data[1] + X_text1_test.tolist()  # text 1
    test_data[2] = test_data[2] + X_text2_test.tolist()  # text 2
    test_data[3] = test_data[3] + y_test.tolist()  # labels

    train_data[0] = train_data[0] + X_pairID_train.tolist()  # pair ids
    train_data[1] = train_data[1] + X_text1_train.tolist()  # text 1
    train_data[2] = train_data[2] + X_text2_train.tolist()  # text 2
    train_data[3] = train_data[3] + y_train.tolist()  # labels

    print(sum(map(lambda x: x == True, train_data[3])))
    print(sum(map(lambda x: x == False, train_data[3])))
    print("---")
    print(sum(map(lambda x: x == True, test_data[3])))
    print(sum(map(lambda x: x == False, test_data[3])))

    print(type(X_pairID_test))
    print(X_pairID_test.shape)
    print(X_pairID_test)
    print(X_text1_train)

    # add to pred result df in the correct order
    print("Appending test split to result dataframe...")
    pred_result_df = pd.concat([pred_result_df, df_dataset_test], ignore_index=True)

# use classifiers
print(f"Finished assembling data. Continue with classification of {str(pred_result_df.shape[0])} examples (train- & test-split)...")
print(f"Train data size: {str(len(train_data[0]))}")
print(f"Test data size: {str(len(test_data[0]))}")

gs_params = {
        'C': range(9, 11),
        'kernel': ('sigmoid', 'rbf')
}

verb, cv, n_jobs = 50, 2, 6

pred_result_df[FASTTEXT] = fasttext_sim(train_data[1], test_data[1],
                                             train_data[2], test_data[2], train_data[3], test_data[3],
                                             gs_params, verb, cv, n_jobs)
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)
pred_result_df[SEM_GLOVE] = semantic_sim_glove(train_data[1], test_data[1],
                                             train_data[2], test_data[2], train_data[3], test_data[3],
                                             gs_params, verb, cv, n_jobs)
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)
pred_result_df[SEM_BERT] = semantic_sim_bert(train_data[1], test_data[1],
                                             train_data[2], test_data[2], train_data[3], test_data[3],
                                             gs_params, verb, cv, n_jobs)
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)
pred_result_df[SEM_T5] = semantic_sim_t5(train_data[1], test_data[1],
                                             train_data[2], test_data[2], train_data[3], test_data[3],
                                             gs_params, verb, cv, n_jobs)
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)
pred_result_df[TFIDF_COSINE] = tfidf_cosine_sim(train_data[1], test_data[1],
                                             train_data[2], test_data[2], train_data[3], test_data[3],
                                             gs_params, verb, cv, n_jobs)
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)
pred_result_df[FUZZY] = fuzzy_sim(train_data[1], test_data[1],
                                             train_data[2], test_data[2], train_data[3], test_data[3],
                                             gs_params, verb, cv, n_jobs)
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)
pred_result_df[NGRAM3] = ngram_sim(3,train_data[1], test_data[1],
                                             train_data[2], test_data[2], train_data[3], test_data[3],
                                             gs_params, verb, cv, n_jobs)
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)



# Output data to json format
print(len(pred_result_df))
pred_result_df = pred_result_df.reset_index(drop=True)
print(len(pred_result_df))
pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                       index=True, indent=4)

print("Done.")
