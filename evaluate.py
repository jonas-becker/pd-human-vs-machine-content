import os
import pandas as pd
from tqdm import tqdm
from re import sub
import numpy as np
from sklearn.utils import shuffle
import re
import sys
from setup import *
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def eval(df, bin_id, threshold, method, eval_df):
    actual = pd.Series(df[PARAPHRASE], name='Actual')
    predicted = pd.Series(df[bin_id], name='Predicted')
    df_cross = pd.crosstab(actual, predicted)
    #print(df_cross)
    f1 = f1_score(actual, predicted)

    # if there were no predicted non-paraphrases insert 0
    try:
        tp = df_cross[1][1]
    except Exception as e:
        tp = 0
        pass
    try:
        tn = df_cross[0][0]
    except Exception as e:
        tn = 0
        pass
    try:
        fp = df_cross[1][0]
    except Exception as e:
        fp = 0
        pass
    try:
        fn = df_cross[0][1]
    except Exception as e:
        fn = 0
        pass
    
    specificity = None
    if (tn+fp) != 0:
        specificity = float(tn / (tn+fp))

    eval_df.loc[len(eval_df.index)] = [
        file.split("_")[0], 
        method, 
        str(len(df)),   # amount of sentence pairs in dataset
        tp, 
        tn, 
        fp, 
        fn, 
        float((tn+tp) / len(df)),   # accuracy
        float(tp / (tp+fp)),   # precision
        float(tp / (tp+fn)),   # recall
        specificity,    # specificity (true negative rate)
        threshold, 
        f1
        ]
    return eval_df

def find_optimal_thresholds(df):
    print("Finding the optimal thresholds for each method...")
    SEM_BERT_THRESHOLD = 0.1
    FUZZY_THRESHOLD = 0.1
    SEM_T5_THRESHOLD = 0.1
    NGRAM_THRESHOLD = 0.1
    TFIDF_THRESHOLD = 0.1
    GPT3_THRESHOLD = 0.1
    sem_bert_best_f1 = 0.0
    fuz_best_f1 = 0.0
    sem_t5_best_f1 = 0.0
    ngram_best_f1 = 0.0
    tfidf_best_f1 = 0.0
    gpt3_best_f1 = 0.0

    for threshold in tqdm(np.round(np.linspace(0,1,10,endpoint=False),2)):
        sem_bert_bin = []
        fuzzy_bin = []
        sem_t5_bin = []
        ngram_bin = []
        tfidf_bin = []
        gpt3_bin = []
        for i, row in df.iterrows():
            # sem_bert
            if row[SEM_BERT] >= threshold:
                sem_bert_bin.append(True)
            else:
                sem_bert_bin.append(False)
            # sem_t5
            if row[SEM_T5] >= threshold:
                sem_t5_bin.append(True)
            else:
                sem_t5_bin.append(False)
            # fuzzy
            if row[FUZZY] >= threshold:
                fuzzy_bin.append(True)
            else:
                fuzzy_bin.append(False)
            # ngram
            if row[NGRAM3] >= threshold:
                ngram_bin.append(True)
            else:
                ngram_bin.append(False)
            # tfidf cosine
            if row[TFIDF_COSINE] >= threshold:
                tfidf_bin.append(True)
            else:
                tfidf_bin.append(False)
            # tfidf cosine
            if row[SEM_GPT3] >= threshold:
                gpt3_bin.append(True)
            else:
                gpt3_bin.append(False)

        df[SEM_BERT_BIN] = sem_bert_bin
        df[SEM_T5_BIN] = sem_t5_bin
        df[FUZZY_BIN] = fuzzy_bin
        df[NGRAM_BIN] = ngram_bin
        df[TFIDF_COSINE_BIN] = tfidf_bin
        df[SEM_GPT3_BIN] = gpt3_bin

        actual = pd.Series(df[PARAPHRASE], name='Actual')

        predicted = pd.Series(df[SEM_BERT_BIN], name='Predicted')
        f1 = f1_score(actual, predicted)
        if f1 > sem_bert_best_f1:
            sem_bert_best_f1 = f1
            SEM_BERT_THRESHOLD = threshold 
            
        predicted = pd.Series(df[SEM_T5_BIN], name='Predicted')
        f1 = f1_score(actual, predicted)
        if f1 > sem_t5_best_f1:
            sem_t5_best_f1 = f1
            SEM_T5_THRESHOLD = threshold 

        predicted = pd.Series(df[FUZZY_BIN], name='Predicted')
        f1 = f1_score(actual, predicted)
        if f1 > fuz_best_f1:
            fuz_best_f1 = f1
            FUZZY_THRESHOLD = threshold 
        
        predicted = pd.Series(df[NGRAM_BIN], name='Predicted')
        f1 = f1_score(actual, predicted)
        if f1 > ngram_best_f1:
            ngram_best_f1 = f1
            NGRAM_THRESHOLD = threshold 
        
        predicted = pd.Series(df[TFIDF_COSINE_BIN], name='Predicted')
        f1 = f1_score(actual, predicted)
        if f1 > tfidf_best_f1:
            tfidf_best_f1 = f1
            TFIDF_THRESHOLD = threshold 
        
        predicted = pd.Series(df[SEM_GPT3_BIN], name='Predicted')
        f1 = f1_score(actual, predicted)
        if f1 > gpt3_best_f1:
            gpt3_best_f1 = f1
            GPT3_THRESHOLD = threshold 
    
    return TFIDF_THRESHOLD, SEM_BERT_THRESHOLD, SEM_T5_THRESHOLD, FUZZY_THRESHOLD, NGRAM_THRESHOLD, GPT3_THRESHOLD
    
c_pal = sns.color_palette("colorblind")
plt.rcParams.update({'font.size': 16})

eval_df = pd.DataFrame(columns=[DATASET_NAME, METHOD, PAIRS, TP, TN, FP, FN, ACCURACY, PRECISION, RECALL, SPECIFICITY ,THRESHOLD, F1])

eval_string = ""
eval_string += "-------------------------" + "\n"
eval_string += "OVERALL DATA STATISTICS" + "\n"
eval_string += "-------------------------" + "\n"
eval_string += "Processed datasets: "
for d in DATASETS:
    eval_string += d + ", "
eval_string = eval_string[:-2] + "\n"


# find the optimal thresholds for each method and dataset and evaluate with said thresholds
for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    if file.split("_")[0] in MACHINE_PARAPHRASED_DATASETS:
        print(f"---> Skipping evaluation of {file} because it is machine-paraphrased and only contains positive pairs...")
        continue
    print(f"---> Evaluating {file}...")
    
    df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")

    # Find the optimal thresholds:
    if len(df[df[PARAPHRASE] == False]) > 10:
        TFIDF_THRESHOLD, SEM_BERT_THRESHOLD, SEM_T5_THRESHOLD, FUZZY_THRESHOLD, NGRAM_THRESHOLD, SEM_GPT3_THRESHOLD = find_optimal_thresholds(df)
    else:
        # if the dataset has only paraphrase-pairs
        TFIDF_THRESHOLD, SEM_BERT_THRESHOLD, SEM_T5_THRESHOLD, FUZZY_THRESHOLD, NGRAM_THRESHOLD, SEM_GPT3_THRESHOLD = 0.7, 0.7, 0.7, 0.7, 0.7, 0.7

    print("Continue with thresholds: ")
    print("sem_bert: " + str(SEM_BERT_THRESHOLD))
    print("sem_t5: " + str(SEM_T5_THRESHOLD))
    print("fuzzy: " + str(FUZZY_THRESHOLD))
    print("ngram: " + str(NGRAM_THRESHOLD))

    # evaluate results with the optimal thresholds:
    sem_bert_bin = []
    sem_t5_bin = []
    fuzzy_bin = []
    ngram_bin = []
    tfidf_bin = []
    gpt3_bin = []
    print("Checking determined thresholds for every pair...")
    for i, row in df.iterrows():
        # sem_bert
        if row[SEM_BERT] >= SEM_BERT_THRESHOLD:
            sem_bert_bin.append(True)
        else:
            sem_bert_bin.append(False)
        # sem_t5
        if row[SEM_T5] >= SEM_T5_THRESHOLD:
            sem_t5_bin.append(True)
        else:
            sem_t5_bin.append(False)
        # fuzzy
        if row[FUZZY] >= FUZZY_THRESHOLD:
            fuzzy_bin.append(True)
        else:
            fuzzy_bin.append(False)
        # ngram
        if row[NGRAM3] >= NGRAM_THRESHOLD:
            ngram_bin.append(True)
        else:
            ngram_bin.append(False)
        # tfidf cosine
        if row[TFIDF_COSINE] >= TFIDF_THRESHOLD:
            tfidf_bin.append(True)
        else:
            tfidf_bin.append(False)
        # sem_gpt3
        if row[SEM_GPT3] >= SEM_GPT3_THRESHOLD:
            gpt3_bin.append(True)
        else:
            gpt3_bin.append(False)

    df[SEM_BERT_BIN] = sem_bert_bin
    df[SEM_T5_BIN] = sem_t5_bin
    df[FUZZY_BIN] = fuzzy_bin
    df[NGRAM_BIN] = ngram_bin
    df[TFIDF_COSINE_BIN] = tfidf_bin
    df[SEM_GPT3_BIN] = gpt3_bin

    print("Calculating precision-recall curves...")
    precision_sem_bert, recall_sem_bert, thresholds_sem_bert = precision_recall_curve(df["is_paraphrase"], df[SEM_BERT])
    precision_sem_t5, recall_sem_t5, thresholds_sem_t5 = precision_recall_curve(df["is_paraphrase"], df[SEM_T5])
    precision_fuzzy, recall_fuzzy, thresholds_fuzzy = precision_recall_curve(df["is_paraphrase"], df[FUZZY])
    precision_tfidf, recall_tfidf, thresholds_tfidf = precision_recall_curve(df["is_paraphrase"], df[TFIDF_COSINE])
    precision_ngram, recall_ngram, thresholds_ngram = precision_recall_curve(df["is_paraphrase"], df[NGRAM3])
    df_gpt3 = df[df[SEM_GPT3].notnull()]
    if df_gpt3.shape[0] != 0:
        precision_gpt3, recall_gpt3, thresholds_gpt3 = precision_recall_curve(df_gpt3["is_paraphrase"], df_gpt3[SEM_GPT3])

    print("Plotting curves...")
    fig, ax = plt.subplots()
    ax.plot(recall_sem_bert, precision_sem_bert, color=c_pal[0], label='BERT')
    ax.plot(recall_sem_t5, precision_sem_t5, color=c_pal[1], label='T5')
    ax.plot(recall_fuzzy, precision_fuzzy, color=c_pal[2], label='Fuzzy')
    ax.plot(recall_tfidf, precision_tfidf, color=c_pal[3], label='TF-IDF')
    ax.plot(recall_ngram, precision_ngram, color=c_pal[4], label='3-Gram')
    if df_gpt3.shape[0] != 0:
        ax.plot(recall_gpt3, precision_gpt3, color=c_pal[5], label='GPT-3')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc = 4)
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, file.split("_")[0]+"_pr.pdf"), bbox_inches='tight')

    # Confusion Matrix and f1 score for the defined threshold
    print("Constructing confusion matrix & calculating f1-score...")

    eval_df = eval(df, SEM_BERT_BIN, SEM_BERT_THRESHOLD, SEM_BERT, eval_df)
    eval_df = eval(df, SEM_T5_BIN, SEM_T5_THRESHOLD, SEM_T5, eval_df)
    eval_df = eval(df, FUZZY_BIN, FUZZY_THRESHOLD, FUZZY, eval_df)
    eval_df = eval(df, TFIDF_COSINE_BIN, TFIDF_THRESHOLD, TFIDF_COSINE, eval_df)
    eval_df = eval(df, NGRAM_BIN, NGRAM_THRESHOLD, NGRAM3, eval_df)
    if df_gpt3.shape[0] != 0:
        eval_df = eval(df, SEM_GPT3_BIN, SEM_GPT3_THRESHOLD, SEM_GPT3, eval_df)

eval_df.to_json(os.path.join(OUT_DIR, EVALUATION_FOLDER, EVALUATION_RESULTS_FILENAME), orient = "index", index = True, indent = 4)
eval_df.to_csv(os.path.join(OUT_DIR, EVALUATION_FOLDER, EVALUATION_RESULTS_FILENAME.replace(".json", ".csv")), index = True)

# Reevaluate all methods & datasets with the same thresholds
'''
eval_df = pd.DataFrame(columns=[DATASET_NAME, METHOD, PAIRS, TP, TN, FP, FN, ACCURACY, PRECISION, RECALL, SPECIFICITY ,THRESHOLD, F1])
for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):    
    for threshold in DEFAULT_THRESHOLDS:
        df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")
        print(f"---> Evaluating {file} with threshold {threshold}...")

        sem_bert_bin = []
        sem_t5_bin = []
        fuzzy_bin = []
        ngram_bin = []
        tfidf_bin = []
        print("Checking determined thresholds for every pair...")
        for i, row in df.iterrows():
            # sem_bert
            if row[SEM_BERT] >= SEM_BERT_THRESHOLD:
                sem_bert_bin.append(True)
            else:
                sem_bert_bin.append(False)
            # sem_t5
            if row[SEM_T5] >= SEM_T5_THRESHOLD:
                sem_t5_bin.append(True)
            else:
                sem_t5_bin.append(False)
            # fuzzy
            if row[FUZZY] >= FUZZY_THRESHOLD:
                fuzzy_bin.append(True)
            else:
                fuzzy_bin.append(False)
            # ngram
            if row[NGRAM3] >= NGRAM_THRESHOLD:
                ngram_bin.append(True)
            else:
                ngram_bin.append(False)
            # tfidf cosine
            if row[TFIDF_COSINE] >= TFIDF_THRESHOLD:
                tfidf_bin.append(True)
            else:
                tfidf_bin.append(False)

        df[SEM_BERT_BIN] = sem_bert_bin
        df[SEM_T5_BIN] = sem_t5_bin
        df[FUZZY_BIN] = fuzzy_bin
        df[NGRAM_BIN] = ngram_bin
        df[TFIDF_COSINE_BIN] = tfidf_bin

        eval_df = eval(df, SEM_BERT_BIN, threshold, SEM_BERT, eval_df)
        eval_df = eval(df, SEM_T5_BIN, threshold, SEM_T5, eval_df)
        eval_df = eval(df, FUZZY_BIN, threshold, FUZZY, eval_df)
        eval_df = eval(df, TFIDF_COSINE_BIN, threshold, TFIDF_COSINE, eval_df)

eval_df.to_json(os.path.join(OUT_DIR, EVALUATION_FOLDER, "thresholds_"+EVALUATION_RESULTS_FILENAME), orient = "index", index = True, indent = 4)
eval_df.to_csv(os.path.join(OUT_DIR, EVALUATION_FOLDER, "thresholds_"+EVALUATION_RESULTS_FILENAME.replace(".json", ".csv")), index = True)
'''

# Make correlation graphs (mixed)
methods_to_correlate = [TFIDF_COSINE, NGRAM3, FUZZY, SEM_BERT, SEM_T5, SEM_GPT3]

# machine-paraphrases & human-paraphrases
df = pd.DataFrame()
for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    tmp_df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")
    df = pd.concat([df,tmp_df])
df = shuffle(df)
corr_df = df.reset_index(drop=True)
# machine-paraphrases
df = pd.DataFrame()
for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    if file.split("_")[0] in MACHINE_PARAPHRASED_DATASETS:
        tmp_df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")
        df = pd.concat([df,tmp_df])
df = shuffle(df)
corr_df_m = df.reset_index(drop=True)
# human-paraphrases
df = pd.DataFrame()
for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    if file.split("_")[0] in HUMAN_PARAPHRASED_DATASETS:
        tmp_df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")
        df = pd.concat([df,tmp_df])
df = shuffle(df)
print(f"Generating correlation graphs for datasets.")
corr_df_h = df.reset_index(drop=True)

for method1 in tqdm(methods_to_correlate):
    for method2 in methods_to_correlate:
        if method1 != method2:
            if method1 == SEM_BERT:
                xlabel = "BERT Cosine Distance"
            elif method1 == SEM_GPT3:
                xlabel = "GPT-3 Cosine Distance"
            elif method1 == SEM_T5:
                xlabel = "T5 Cosine Distance"
            elif method1 == FUZZY:
                xlabel = "Fuzzy Similarity"
            elif method1 == TFIDF_COSINE:
                xlabel = "TF-IDF Cosine Distance"
            elif method1 == NGRAM3:
                xlabel = "3-Gram Similarity"
            if method2 == SEM_BERT:
                ylabel = "BERT Cosine Distance"
            elif method2 == SEM_GPT3:
                ylabel = "GPT-3 Cosine Distance"
            elif method2 == SEM_T5:
                ylabel = "T5 Cosine Distance"
            elif method2 == FUZZY:
                ylabel = "Fuzzy Similarity"
            elif method2 == TFIDF_COSINE:
                ylabel = "TF-IDF Cosine Distance"
            elif method2 == NGRAM3:
                ylabel = "3-Gram Similarity"

            this_corr_df_m = corr_df_m[corr_df_m[method1].notnull() & corr_df_m[method2].notnull()].truncate(after=CORR_GRAPH_SIZE)
            this_corr_df_h = corr_df_h[corr_df_h[method1].notnull() & corr_df_h[method2].notnull()].truncate(after=CORR_GRAPH_SIZE)

            plt.clf()
            sns.regplot(x=this_corr_df_m[method1], y=this_corr_df_m[method2], scatter_kws={"color": c_pal[0]}, line_kws={"color": c_pal[3]})
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "mixed", method1 + "_" + method2 + "_machine.pdf"), bbox_inches='tight')
            plt.clf()
            sns.regplot(x=this_corr_df_h[method1], y=this_corr_df_h[method2], scatter_kws={"color": c_pal[0]}, line_kws={"color": c_pal[3]})
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "mixed", method1 + "_" + method2 + "_human.pdf"), bbox_inches='tight')

# Make correlation graphs (paraphrases only)
methods_to_correlate = [TFIDF_COSINE, NGRAM3, FUZZY, SEM_BERT, SEM_T5, SEM_GPT3]

# machine-paraphrases & human-paraphrases
df = pd.DataFrame()
for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    tmp_df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")
    df = pd.concat([df,tmp_df])
df = shuffle(df)
corr_df = df[df[PARAPHRASE] == True].reset_index(drop=True)
# machine-paraphrases
df = pd.DataFrame()
for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    if file.split("_")[0] in MACHINE_PARAPHRASED_DATASETS:
        tmp_df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")
        df = pd.concat([df,tmp_df])
df = shuffle(df)
corr_df_m = df[df[PARAPHRASE] == True].reset_index(drop=True)
# human-paraphrases
df = pd.DataFrame()
for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    if file.split("_")[0] in HUMAN_PARAPHRASED_DATASETS:
        tmp_df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")
        df = pd.concat([df,tmp_df])
df = shuffle(df)
print(f"Generating correlation graphs for datasets.")
corr_df_h = df[df[PARAPHRASE] == True].reset_index(drop=True)

for method1 in tqdm(methods_to_correlate):
    for method2 in methods_to_correlate:
        if method1 != method2:
            if method1 == SEM_BERT:
                xlabel = "BERT Cosine Distance"
            elif method1 == SEM_GPT3:
                xlabel = "GPT-3 Cosine Distance"
            elif method1 == SEM_T5:
                xlabel = "T5 Cosine Distance"
            elif method1 == FUZZY:
                xlabel = "Fuzzy Similarity"
            elif method1 == TFIDF_COSINE:
                xlabel = "TF-IDF Cosine Distance"
            elif method1 == NGRAM3:
                xlabel = "3-Gram Similarity"
            if method2 == SEM_BERT:
                ylabel = "BERT Cosine Distance"
            elif method2 == SEM_GPT3:
                ylabel = "GPT-3 Cosine Distance"
            elif method2 == SEM_T5:
                ylabel = "T5 Cosine Distance"
            elif method2 == FUZZY:
                ylabel = "Fuzzy Similarity"
            elif method2 == TFIDF_COSINE:
                ylabel = "TF-IDF Cosine Distance"
            elif method2 == NGRAM3:
                ylabel = "3-Gram Similarity"

            this_corr_df_m = corr_df_m[corr_df_m[method1].notnull() & corr_df_m[method2].notnull()].truncate(after=CORR_GRAPH_SIZE)
            this_corr_df_h = corr_df_h[corr_df_h[method1].notnull() & corr_df_h[method2].notnull()].truncate(after=CORR_GRAPH_SIZE)

            plt.clf()
            sns.regplot(x=this_corr_df_m[method1], y=this_corr_df_m[method2], scatter_kws={"color": c_pal[0]}, line_kws={"color": c_pal[3]})
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "paraphrases_only", method1 + "_" + method2 + "_machine.pdf"), bbox_inches='tight')
            plt.clf()
            sns.regplot(x=this_corr_df_h[method1], y=this_corr_df_h[method2], scatter_kws={"color": c_pal[0]}, line_kws={"color": c_pal[3]})
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "paraphrases_only", method1 + "_" + method2 + "_human.pdf"), bbox_inches='tight')

# Generate Correlation Matrix & Output Heatmap
plt.clf()
corr_h_matrix = corr_df_h.drop(columns=[PARAPHRASE, COSINE_DISTANCE]).corr()
mask = np.triu(np.ones_like(corr_h_matrix, dtype=bool)) # Generate a mask for the upper triangle
sns.heatmap(corr_h_matrix, mask=mask, annot=True, xticklabels=["TF-IDF", "3-Gram", "Fuzzy", "GPT-3", "BERT", "T5"], yticklabels=["TF-IDF", "3-Gram", "Fuzzy", "GPT-3", "BERT", "T5"], vmin=-1, vmax=1)
plt.yticks(rotation=0)
plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_human.pdf"), bbox_inches="tight")

plt.clf()
corr_m_matrix = corr_df_m.drop(columns=[PARAPHRASE, COSINE_DISTANCE]).corr()
sns.heatmap(corr_m_matrix, mask=mask, annot=True, xticklabels=["TF-IDF", "3-Gram", "Fuzzy", "GPT-3", "BERT", "T5"], yticklabels=["TF-IDF", "3-Gram", "Fuzzy", "GPT-3", "BERT", "T5"], vmin=-1, vmax=1)   #, xticklabels=None, yticklabels=None
plt.yticks(rotation=0)
plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_machine.pdf"), bbox_inches="tight")
plt.clf()

# Generate Correlation Matrix & Output Heatmap (without NGram)
plt.clf()
corr_h_matrix = corr_df_h.drop(columns=[PARAPHRASE, COSINE_DISTANCE, NGRAM3]).corr()
mask = np.triu(np.ones_like(corr_h_matrix, dtype=bool)) # Generate a mask for the upper triangle
sns.heatmap(corr_h_matrix, mask=mask, annot=True, xticklabels=["TF-IDF", "Fuzzy", "GPT-3", "BERT", "T5"], yticklabels=["TF-IDF", "Fuzzy", "GPT-3", "BERT", "T5"], vmin=-1, vmax=1)
plt.yticks(rotation=0)
plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_human_nongram.pdf"), bbox_inches="tight")

plt.clf()
corr_m_matrix = corr_df_m.drop(columns=[PARAPHRASE, COSINE_DISTANCE, NGRAM3]).corr()
sns.heatmap(corr_m_matrix, mask=mask, annot=True, xticklabels=["TF-IDF", "Fuzzy", "GPT-3", "BERT", "T5"], yticklabels=["TF-IDF", "Fuzzy", "GPT-3", "BERT", "T5"], vmin=-1, vmax=1)   #, xticklabels=None, yticklabels=None
plt.yticks(rotation=0)
plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_machine_nongram.pdf"), bbox_inches="tight")
plt.clf()

print("Done.")
