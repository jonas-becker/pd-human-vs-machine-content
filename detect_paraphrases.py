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
import numpy
from sklearn.metrics import roc_auc_score
import io
import matplotlib.pyplot as plt
import pprint
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import PrecisionRecallDisplay
from IPython.display import display
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import zipfile
import tensorflow as tf
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import tokenization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import math

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_bert_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    sequence_output = bert_layer({"input_word_ids": input_word_ids, "input_mask": input_mask, "input_type_ids": segment_ids})["sequence_output"]
    print(sequence_output)
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def GridSearch_table_plot(grid_clf, param_name,
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

    scores_df.to_csv(os.path.join(OUT_DIR, DETECTION_FOLDER, "grid_search.csv"))

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
        plt.show()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def preprocess_function_text1(df):
    return bert_tokenizer(df[TEXT1], truncation=True)


def preprocess_function_text2(df):
    return bert_tokenizer(df[TEXT2], truncation=True)


def append_result(df, test_result_df, test_pair_ids, test_pred_result):
    test_result_df = pd.concat(test_result_df)

    return test_result_df


def semantic_sim_bert_new(pair_ids_train, pair_ids_test, text1_train, text1_test, text2_train, text2_test, y_train, y_test):
    print("Semantic Similarity (BERT) \n------------")
    print("Loading module from tensorflow...")
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4"
    print("Creating BERT layer...")
    bert_layer = hub.KerasLayer(module_url, trainable=True)

    print("Initialize BERT layer settings...")
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = bert_tokenizer

    print("Create BERT embeddings (train)...")
    train_input = tokenizer(text1_train, text2_train, padding="max_length", truncation=True).data
    print(type(train_input))
    print("Create BERT embeddings (test)...")
    test_input = tokenizer(text1_test, text2_test, padding="max_length", truncation=True).data

    print(len(text1_test))
    print(len(test_input["attention_mask"]))

    print("Building BERT model...")
    model = build_bert_model(bert_layer, max_len=512)
    model.summary()

    logf = open("error.log", "w")

    print("Transfer training the BERT model...")
    try:
        train_history = model.fit(
            train_input, y_train,
            epochs=1,
            batch_size=16
        )
    except Exception as e:  # most generic exception you can catch
        logf.write(str(e))

    print("Saving model...")
    model.save('models/bert.h5')

    # get output of slice layer from model above

    print("Initializing SVM Model...")
    cls_layer_model = Model(model.input, outputs=model.get_layer('tf.__operators__.getitem').output)

    print("Get BERT predictions...")
    print(train_input.keys())
    X_train = cls_layer_model.predict({"input_word_ids": np.array(train_input["input_ids"]), "input_mask": np.array(train_input["attention_mask"]), "segment_ids": np.array(train_input["token_type_ids"])})
    X_test = cls_layer_model.predict({"input_word_ids": np.array(test_input["input_ids"]), "input_mask": np.array(test_input["attention_mask"]), "segment_ids": np.array(test_input["token_type_ids"])})

    print(len(X_test))

    print("Training the SVM...")
    model_svm = SVC(C=15, kernel='rbf', gamma=0.001, probability=True)
    model_svm.fit(X_train, y_train)

    print("Predicting test split...")
    prediction_result = model_svm.predict_proba(X_test)
    print(prediction_result)
    print(len(prediction_result))
    print(prediction_result.shape)
    return [p[1] for p in prediction_result]    # only get probability for one of two classes (true/false)


def semantic_sim_bert(pair_ids_train, pair_ids_test, text1_train, text1_test, text2_train, text2_test, y_train, y_test):
    print("Calculating semantic similarity with BERT.")
    print("Train examples: " + str(len(pair_ids_train)))
    print("Test examples: " + str(len(pair_ids_test)))
    model = SentenceTransformer('all-distilroberta-v1')
    print("Created model.")

    print(sum(map(lambda x: x is True, y_test)))
    print(sum(map(lambda x: x is False, y_test)))

    print("Creating embeddings (test split).")
    X1_test = model.encode(text1_test)
    #print(X1_test.shape)
    #print(X1_test[0])
    X2_test = model.encode(text2_test)
    #print(X2_test.shape)
    #print(X2_test[0])
    X_test = np.column_stack((X1_test, X2_test))
    #print(X_test[0])
    #print(X_test.shape)
    print("Creating embeddings (train split).")
    X1_train = model.encode(text1_train)
    X2_train = model.encode(text2_train)
    X_train = np.column_stack((X1_train, X2_train))

    #scaler = StandardScaler()
    #print(np.concatenate((X1_test,X2_test,X1_train,X2_train)).shape)
    #scaler.fit( np.concatenate((X1_test,X2_test,X1_train,X2_train)))
    #print(scaler.transform(X1_test[0]))
    
    '''
    print("Creating embeddings for text_2s (test).")
    X2_test = model.encode(text2_test)
    X_test = np.column_stack((X1_test, X2_test))

    print("Creating embeddings for text_1s (training).")
    X1_train = model.encode(text1_train)
    print("Creating embeddings for text_2s (training).")
    X2_train = model.encode(text2_train)
    X_train = np.column_stack((X1_train, X2_train))
    '''

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
    #clf = make_pipeline(
    #    SGDClassifier(max_iter=1000, class_weight={False: class_weight[0], True: class_weight[1]},
    #                  verbose=True, loss="modified_huber")
    #                    )

    # Grid Search
    clf = svm.SVC()
    gs_params = {
        'C': range(9, 11),
        'kernel': ('sigmoid', 'rbf')
    }
    gs = GridSearchCV(clf, gs_params, cv=5, verbose=10)

    # train the model
    print("Training the SVM and calculating decision function scores for the test split...")

    gs.fit(X_train, y_train)
    # use the model to predict the testing instances
    print("Testing the SVM...")
    y_pred = gs.predict(X_test)
    # generate the classification report
    print(classification_report(y_test, y_pred))

    y_score = gs.decision_function(X_test)

    prediction_result = [sigmoid(y_item) for y_item in y_score.tolist()]  # normalize with sigmoid

    GridSearch_table_plot(gs, "C", negative=False)

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

print(None in df[TEXT2].tolist())
print(None in df[TEXT1].tolist())

pred_result_df = pd.DataFrame()

df = df.reset_index(drop=True)
df = df.truncate(before=0, after=10000)     # for testing
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

    print("Creating splits...")
    X_pairID_train, X_pairID_test, X_text1_train, X_text1_test, X_text2_train, X_text2_test, y_train, y_test = train_test_split(df_dataset[PAIR_ID], df_dataset[TEXT1], df_dataset[TEXT2], y, test_size=0.3, random_state=0, stratify=y)

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

    # add only to df if pair is not included already, this does not include results yet
    print(type(X_pairID_test))
    print(X_pairID_test.shape)
    print(X_pairID_test)
    print(X_text1_train)
    pred_result_df = pd.concat([pred_result_df, df_dataset[df_dataset[PAIR_ID].isin(X_pairID_test)]], ignore_index=True)

    if None in train_data[2]:
        print("NOIKFIOSJFIUJHSAIUJDHAIUHDIUOSA")
        break

# use classifiers
print(f"Finished assembling data. Continue with classification of {str(pred_result_df.shape[0])} examples (train- & test-split)...")

pred_result_df[SEM_BERT] = semantic_sim_bert_new(train_data[0], test_data[0], train_data[1], test_data[1],
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
