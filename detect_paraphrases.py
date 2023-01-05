import numpy as np
# patch for intel speedup:
from sklearnex import patch_sklearn
patch_sklearn()
import pandas as pd
from sklearn.svm import SVC
from tqdm import tqdm
from thefuzz import fuzz
from setup import *
from strsimpy.ngram import NGram
from gensim.models import fasttext
import numpy
from torch import nn
import matplotlib.pyplot as plt
import pprint
from transformers import BertModel, T5Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display
import fasttext
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from tensorflow.python.ops.numpy_ops import np_config
import gc
import joblib
gc.collect()

np_config.enable_numpy_behavior()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device " + str(device))
torch.cuda.empty_cache()


def save_gridsearch_model(gs, filename):
    """
        Saves a model (gridsearch model) to a file. It can later be laded using joblib.load(filename).
        :param gs: the GridSearchCV model
        :param filename: the desired filename
        :return:
    """
    if not os.path.exists(os.path.join(MODELS_FOLDER, GRIDSEARCH_FOLDER)):
        os.makedirs(os.path.join(MODELS_FOLDER, GRIDSEARCH_FOLDER))
        
    joblib.dump(gs, os.path.join(MODELS_FOLDER, GRIDSEARCH_FOLDER, filename+'.pkl'))
    print("Saved model to: " + os.path.join(MODELS_FOLDER, GRIDSEARCH_FOLDER, filename+'.pkl'))


def gridsearch_table_plot(grid_clf, param_name, method_name, num_results=15, negative=True, graph=True,
                          display_all_params=True):
    """
        Display grid search results in a figure
        :param grid_clf: the estimator resulting from a grid search
                           for example: grid_clf = GridSearchCV( ...
        :param param_name: a string with the name of the parameter being tested
        :param num_results: an integer indicating the number of results to display
                           Default: 15
        :param negative: boolean: should the sign of the score be reversed?
                           scoring = 'neg_log_loss', for instance
                           Default: True
        :param graph: boolean: should a graph be produced?
                           non-numeric parameters (True/False, None) don't graph well
                           Default: True
        :param display_all_params: boolean: should we print out of all the parameters, not just the ones searched for?
                           Default: True
    """

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
    display(pd.DataFrame(cv_results).sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    if not os.path.exists(os.path.join(OUT_DIR, DETECTION_FOLDER, GRIDSEARCH_FOLDER)):
        os.makedirs(os.path.join(OUT_DIR, DETECTION_FOLDER, GRIDSEARCH_FOLDER))

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
        plt.savefig(os.path.join(OUT_DIR, DETECTION_FOLDER, GRIDSEARCH_FOLDER, 'gs_' + method_name + '.png'))

    # save stats
    scores_df.sort_index(inplace=True)
    scores_df.to_csv(os.path.join(OUT_DIR, DETECTION_FOLDER, GRIDSEARCH_FOLDER, "gs_" + method_name + ".csv"))


def predict_with_model(model, gs, X_test):
    """
        Test a provided model by providing an input
        :param model: the model to test
        :param gs: the gridsearch object including the model
        :param X_test: the test features
        :return: estimated probabilities for being a paraphrase, predicted labels
    """
    print("Predicting...")
    prediction_classes = model.predict(X_test).tolist()
    prediction_result = model.predict_proba(X_test)
    true_i = numpy.where(gs.classes_ == True)[0][0]  # the index of the prob for being "True"

    # only get probability for one of two classes (true/false), True/False classification
    return [p[true_i] for p in prediction_result], prediction_classes


def semantic_sim_bert(text1_train, text1_test, text2_train, text2_test, y_train, gs_params, verb, cv, n_jobs, max_gs_iter):
    """
        Calculates the BERT embeddings for the given text pairs and uses an SVM to output classification
        and estimated probabilities for the provided test split. The optimal SVM parameters are determined by a grid
        search
        :param text1_train: first texts of train split
        :param text1_test: first texts of test split
        :param text2_train: second texts of train split
        :param text2_test: second texts of test split
        :param y_train: true label for the train split
        :param gs_params: grid search parameters
        :param verb: the verbose level (amount of logging)
        :param cv: how many folds should be done during grid search
        :param n_jobs: how many cpu cores to use for processing (-1 for max amount)
        :return: estimated probabilities for being a paraphrase, predicted labels
    """
    print("Semantic Similarity (BERT) \n------------")
    print("Loading model...")
    model = BertModel.from_pretrained("bert-large-uncased").to(device)
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')

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

    print(f'Amount of features per pair: {X_train.shape[1]}')

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    # Grid Search
    gs = RandomizedSearchCV(model_svm, gs_params, n_iter=max_gs_iter, cv=cv, verbose=verb, n_jobs=n_jobs, random_state=0)
    gs.fit(X_train, y_train)
    gridsearch_table_plot(gs, "C", "BERT", negative=False)

    save_gridsearch_model(gs, "bert_gs")
    best_model = gs.best_estimator_

    return predict_with_model(best_model, gs, X_test)


def semantic_sim_t5(text1_train, text1_test, text2_train, text2_test, y_train, gs_params, verb, cv, n_jobs, max_gs_iter):
    """
        Calculates the T5 embeddings for the given text pairs and uses an SVM to output classification
        and estimated probabilities for the provided test split. The optimal SVM parameters are determined by a grid
        search
        :param text1_train: first texts of train split
        :param text1_test: first texts of test split
        :param text2_train: second texts of train split
        :param text2_test: second texts of test split
        :param y_train: true label for the train split
        :param gs_params: grid search parameters
        :param verb: the verbose level (amount of logging)
        :param cv: how many folds should be done during grid search
        :param n_jobs: how many cpu cores to use for processing (-1 for max amount)
        :return: estimated probabilities for being a paraphrase, predicted labels
    """
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

    print(f'Amount of features per pair: {X_train.shape[1]}')

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    # Grid Search
    gs = RandomizedSearchCV(model_svm, gs_params, n_iter=max_gs_iter, cv=cv, verbose=verb, n_jobs=n_jobs, random_state=0)
    gs.fit(X_train, y_train)
    gridsearch_table_plot(gs, "C", "T5", negative=False)

    save_gridsearch_model(gs, "t5_gs")
    best_model = gs.best_estimator_

    return predict_with_model(best_model, gs, X_test)


def ngram_sim(n, text1_train, text1_test, text2_train, text2_test, y_train, gs_params, verb, cv, n_jobs, max_gs_iter):
    """
        Calculates the N-Gram similarities for the given text pairs and uses an SVM to output classification
        and estimated probabilities for the provided test split. The optimal SVM parameters are determined by a grid
        search.
        Done after http://webdocs.cs.ualberta.ca/~kondrak/papers/spire05.pdf
        :param n: n-gram span
        :param text1_train: first texts of train split
        :param text1_test: first texts of test split
        :param text2_train: second texts of train split
        :param text2_test: second texts of test split
        :param y_train: true label for the train split
        :param gs_params: grid search parameters
        :param verb: the verbose level (amount of logging)
        :param cv: how many folds should be done during grid search
        :param n_jobs: how many cpu cores to use for processing (-1 for max amount)
        :return: estimated probabilities for being a paraphrase, predicted labels
    """
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

    print(f'Amount of features per pair: {len(sims_train[0])}')

    # Grid Search
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    gs = RandomizedSearchCV(model_svm, gs_params, n_iter=max_gs_iter, cv=cv, verbose=verb, n_jobs=n_jobs, random_state=0)

    # train the model
    print("Training the SVM...")
    gs.fit(sims_train, y_train)
    print("Output grid search stats...")
    gridsearch_table_plot(gs, "C", "ngram", negative=False)

    # use the model to predict the testing instances
    save_gridsearch_model(gs, "ngram_gs")
    best_model = gs.best_estimator_

    return predict_with_model(best_model, gs, sims_test)


def fuzzy_sim(text1_train, text1_test, text2_train, text2_test, y_train, gs_params, verb, cv, n_jobs, max_gs_iter):
    """
        Calculates the fuzzy similarities for the given text pairs and uses an SVM to output classification
        and estimated probabilities for the provided test split. The optimal SVM parameters are determined by a grid
        search
        :param text1_train: first texts of train split
        :param text1_test: first texts of test split
        :param text2_train: second texts of train split
        :param text2_test: second texts of test split
        :param y_train: true label for the train split
        :param gs_params: grid search parameters
        :param verb: the verbose level (amount of logging)
        :param cv: how many folds should be done during grid search
        :param n_jobs: how many cpu cores to use for processing (-1 for max amount)
        :return: estimated probabilities for being a paraphrase, predicted labels
    """
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

    print(f'Amount of features per pair: {len(sims_train[0])}')

    # Grid Search
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    gs = RandomizedSearchCV(model_svm, gs_params, n_iter=max_gs_iter, cv=cv, verbose=verb, n_jobs=n_jobs, random_state=0)

    # train the model
    print("Training the SVM...")
    gs.fit(sims_train, y_train)
    print("Output grid search stats...")
    gridsearch_table_plot(gs, "C", "fuzzy", negative=False)

    save_gridsearch_model(gs, "fuzzy_gs")
    best_model = gs.best_estimator_

    return predict_with_model(best_model, gs, sims_test)


def create_glove_embedding_matrix(word_index, embedding_dict, dimension):
    """
        Creates the embedding matrix for GloVe
        :param word_index: word index of the tokenizer
        :param embedding_dict: the GloVe embedding dictionary
        :param dimension: desired dimensionality of the embedding matrix
        :return: the embedding matrix
    """
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))

    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix


def semantic_sim_glove(text1_train, text1_test, text2_train, text2_test, y_train, gs_params, verb, cv, n_jobs, max_gs_iter):
    """
        Calculates the GloVe embeddings for the given text pairs and uses an SVM to output classification
        and estimated probabilities for the provided test split. The optimal SVM parameters are determined by a grid
        search
        :param text1_train: first texts of train split
        :param text1_test: first texts of test split
        :param text2_train: second texts of train split
        :param text2_test: second texts of test split
        :param y_train: true label for the train split
        :param gs_params: grid search parameters
        :param verb: the verbose level (amount of logging)
        :param cv: how many folds should be done during grid search
        :param n_jobs: how many cpu cores to use for processing (-1 for max amount)
        :return: estimated probabilities for being a paraphrase, predicted labels
    """
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
    embedding_matrix = create_glove_embedding_matrix(tokenizer.word_index, embedding_dict=glove_embedding, dimension=100)

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

    print(f'Amount of features per pair: {X_train.shape[1]}')

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    # Grid Search
    gs = RandomizedSearchCV(model_svm, gs_params, n_iter=max_gs_iter, cv=cv, verbose=verb, n_jobs=n_jobs, random_state=0)
    gs.fit(X_train, y_train)
    gridsearch_table_plot(gs, "C", "GloVe", negative=False)

    save_gridsearch_model(gs, "glove_gs")
    best_model = gs.best_estimator_

    return predict_with_model(best_model, gs, X_test)


def fasttext_sim(text1_train, text1_test, text2_train, text2_test, y_train, gs_params, verb, cv, n_jobs, max_gs_iter):
    """
        Calculates the FastText vector representations for the given text pairs and uses an SVM to output classification
        and estimated probabilities for the provided test split. The optimal SVM parameters are determined by a grid
        search
        :param text1_train: first texts of train split
        :param text1_test: first texts of test split
        :param text2_train: second texts of train split
        :param text2_test: second texts of test split
        :param y_train: true label for the train split
        :param gs_params: grid search parameters
        :param verb: the verbose level (amount of logging)
        :param cv: how many folds should be done during grid search
        :param n_jobs: how many cpu cores to use for processing (-1 for max amount)
        :return: estimated probabilities for being a paraphrase, predicted labels
    """
    print("FastText Similarity \n------------")
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

    print(f'Amount of features per pair: {X_train.shape[1]}')

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    # Grid Search
    gs = RandomizedSearchCV(model_svm, gs_params, n_iter=max_gs_iter, cv=cv, verbose=verb, n_jobs=n_jobs, random_state=0)
    gs.fit(X_train, y_train)
    gridsearch_table_plot(gs, "C", "fasttext", negative=False)

    save_gridsearch_model(gs, "fasttext_gs")
    best_model = gs.best_estimator_

    return predict_with_model(best_model, gs, X_test)


def tfidf_cosine_sim(text1_train, text1_test, text2_train, text2_test, y_train, gs_params, verb, cv, n_jobs, max_gs_iter):
    """
        Calculates the tfidf-vector representations for the given text pairs to calculate cosine similarites of all
        pairs.
        It uses an SVM to output classification and estimated probabilities for the provided test split.
        The optimal SVM parameters are determined by a grid search
        :param text1_train: first texts of train split
        :param text1_test: first texts of test split
        :param text2_train: second texts of train split
        :param text2_test: second texts of test split
        :param y_train: true label for the train split
        :param gs_params: grid search parameters
        :param verb: the verbose level (amount of logging)
        :param cv: how many folds should be done during grid search
        :param n_jobs: how many cpu cores to use for processing (-1 for max amount)
        :return: estimated probabilities for being a paraphrase, predicted label
    """
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

    print(f'Amount of features per pair: {len(sims_train[0])}')

    # Grid Search
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    gs = RandomizedSearchCV(model_svm, gs_params, n_iter=max_gs_iter, cv=cv, verbose=verb, n_jobs=n_jobs, random_state=0)

    # train the model
    print("Training the SVM...")
    gs.fit(sims_train, y_train)
    print("Output grid search stats...")
    gridsearch_table_plot(gs, "C", "tfidf_cosine", negative=False)

    save_gridsearch_model(gs, "tfidf_cosine_gs")
    best_model = gs.best_estimator_

    return predict_with_model(best_model, gs, sims_test)


if __name__ == "__main__":

    # create output folder
    if not os.path.isdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
        os.makedirs(os.path.join(OUT_DIR, DETECTION_FOLDER))

    stats_str = "STATISTICS OF DETECTION SCRIPT\n\n"

    print("Reading " + FORMATTED_DATA_FILENAME + " ...")
    df = pd.read_json(os.path.join(OUT_DIR, FORMATTED_DATA_FILENAME), orient="index")
    df = df.sort_values(by=[DATASET])  # sort to have datasets processed sequentially

    print(None in df[TEXT2].tolist())
    print(None in df[TEXT1].tolist())

    pred_result_df = pd.DataFrame()

    #df = df.reset_index(drop=True)
    #df = df.truncate(before=0, after=1000)     # for testing
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

    # Grid Search Space (https://arxiv.org/abs/2101.09023)
    gs_params = {
        'kernel': ('linear', 'rbf', 'poly'),
        'gamma': [0.01, 0.001, 0.0001, 0.00001],
        'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'C': [1, 10, 100],
    }

    verb, cv, n_jobs, max_gs_iter = 50, 3, -1, 40

    pred_result_df[TFIDF_COSINE], pred_result_df[TFIDF_COSINE_PRED] = tfidf_cosine_sim(train_data[1], test_data[1],
                                                                                       train_data[2], test_data[2],
                                                                                       train_data[3],
                                                                                       gs_params, verb, cv, n_jobs, max_gs_iter)
    pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                           index=True, indent=4)
    pred_result_df[FASTTEXT], pred_result_df[FASTTEXT_PRED] = fasttext_sim(train_data[1], test_data[1],
                                                 train_data[2], test_data[2], train_data[3],
                                                 gs_params, verb, cv, n_jobs, max_gs_iter)
    pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                           index=True, indent=4)
    pred_result_df[SEM_GLOVE], pred_result_df[SEM_GLOVE_PRED] = semantic_sim_glove(train_data[1], test_data[1],
                                                 train_data[2], test_data[2], train_data[3],
                                                 gs_params, verb, cv, n_jobs, max_gs_iter)
    pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                           index=True, indent=4)
    pred_result_df[SEM_BERT], pred_result_df[SEM_BERT_PRED] = semantic_sim_bert(train_data[1], test_data[1],
                                                                                train_data[2], test_data[2],
                                                                                train_data[3], gs_params, verb, cv,
                                                                                n_jobs, max_gs_iter)
    pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                           index=True, indent=4)
    pred_result_df[SEM_T5], pred_result_df[SEM_T5_PRED] = semantic_sim_t5(train_data[1], test_data[1],
                                                 train_data[2], test_data[2], train_data[3],
                                                 gs_params, verb, cv, n_jobs, max_gs_iter)
    pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                           index=True, indent=4)
    pred_result_df[FUZZY], pred_result_df[FUZZY_PRED] = fuzzy_sim(train_data[1], test_data[1],
                                                 train_data[2], test_data[2], train_data[3],
                                                 gs_params, verb, cv, n_jobs, max_gs_iter)
    pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                           index=True, indent=4)
    pred_result_df[NGRAM3], pred_result_df[NGRAM3_PRED] = ngram_sim(3,train_data[1], test_data[1],
                                                 train_data[2], test_data[2], train_data[3],
                                                 gs_params, verb, cv, n_jobs, max_gs_iter)
    pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                           index=True, indent=4)



    # Output data to json format
    print("Output results to specified directory...")
    pred_result_df = pred_result_df.reset_index(drop=True)
    pred_result_df.to_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index",
                           index=True, indent=4)

    print("Done.")
