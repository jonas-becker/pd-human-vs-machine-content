import os
import pandas as pd
from tqdm import tqdm
from re import sub
import numpy as np
import re
import sys
from setup import *
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

eval_df = pd.DataFrame(columns=[DATASET_NAME, METHOD, PAIRS, TP, TN, FP, FN, ACCURACY, PRECISION, RECALL, THRESHOLD, F1])

eval_string = ""
eval_string += "-------------------------" + "\n"
eval_string += "OVERALL DATA STATISTICS" + "\n"
eval_string += "-------------------------" + "\n"
eval_string += "Processed datasets: "
for d in DATASETS:
    eval_string += d + ", "
eval_string = eval_string[:-2] + "\n"


# use etpc for threshold choosing as it is a very balanced and diverse dataset
threshold_df =  pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "APH_result.json"), orient = "index")
threshold_df = threshold_df[(threshold_df[TEXT1] != "") & (threshold_df[TEXT2] != "")].reset_index(drop=True)

for file in os.listdir(os.path.join(OUT_DIR, DETECTION_FOLDER)):
    print(f"---> Evaluating {file}...")
    df = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, file), orient = "index")

    # Find the optimal thresholds:
    SEMANTIC_THRESHOLD = 0.1
    FUZZY_THRESHOLD = 0.1
    sem_best_f1 = 0.0
    fuz_best_f1 = 0.0

    print("Finding the optimal thresholds for this dataset...")
    for threshold in tqdm(np.round(np.linspace(0,1,100,endpoint=False),2)):
        semantic_bin = []
        fuzzy_bin = []
        for i, row in df.iterrows():
            # semantic
            if row[SEMANTIC] >= threshold:
                semantic_bin.append(True)
            else:
                semantic_bin.append(False)
            # fuzzy
            if row[FUZZY] >= threshold:
                fuzzy_bin.append(True)
            else:
                fuzzy_bin.append(False)

        df[SEMANTIC_BIN] = semantic_bin
        df[FUZZY_BIN] = fuzzy_bin

        actual = pd.Series(df["is_paraphrase"], name='Actual')
        predicted = pd.Series(df[SEMANTIC_BIN], name='Predicted')
        f1 = f1_score(actual, predicted)
        if f1 > sem_best_f1:
            sem_best_f1 = f1
            SEMANTIC_THRESHOLD = threshold 

        actual = pd.Series(df["is_paraphrase"], name='Actual')
        predicted = pd.Series(df[FUZZY_BIN], name='Predicted')
        f1 = f1_score(actual, predicted)
        if f1 > sem_best_f1:
            fuz_best_f1 = f1
            FUZZY_THRESHOLD = threshold 

    print("Continue with thresholds: ")
    print("Semantic: " + str(SEMANTIC_THRESHOLD))
    print("Fuzzy: " + str(FUZZY_THRESHOLD))

    semantic_bin = []
    fuzzy_bin = []
    print("Checking defined thresholds for every pair...")
    for i, row in df.iterrows():
        # semantic
        if row[SEMANTIC] >= SEMANTIC_THRESHOLD:
            semantic_bin.append(True)
        else:
            semantic_bin.append(False)
        # fuzzy
        if row[FUZZY] >= FUZZY_THRESHOLD:
            fuzzy_bin.append(True)
        else:
            fuzzy_bin.append(False)

    df[SEMANTIC_BIN] = semantic_bin
    df[FUZZY_BIN] = fuzzy_bin

    print("Calculating precision-recall curves...")
    precision_semantic, recall_semantic, thresholds_semantic = precision_recall_curve(df["is_paraphrase"], df[SEMANTIC])
    precision_fuzzy, recall_fuzzy, thresholds_fuzzy = precision_recall_curve(df["is_paraphrase"], df[FUZZY])

    print("Plotting curves...")
    fig, ax = plt.subplots()
    ax.plot(recall_semantic, precision_semantic, color='blue')
    ax.set_title('Precision-Recall Curve (Semantic)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, file.split("_")[0]+"_pr_semantic"))

    fig, ax = plt.subplots()
    ax.plot(recall_fuzzy, precision_fuzzy, color='blue')
    ax.set_title('Precision-Recall Curve (Fuzzy)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, file.split("_")[0]+"_pr_fuzzy"))

    # Confusion Matrix and f1 score for the defined threshold
    print("Constructing confusion matrix & calculating f1-score...")
    
    actual = pd.Series(df["is_paraphrase"], name='Actual')
    predicted = pd.Series(df[SEMANTIC_BIN], name='Predicted')
    df_cross = pd.crosstab(actual, predicted)
    f1 = f1_score(actual, predicted)

    tp = df_cross[1][1]
    tn = df_cross[0][0]
    fp = df_cross[1][0]
    fn = df_cross[0][1]
    eval_df.loc[len(eval_df.index)] = [
        file.split("_")[0], 
        "semantic", 
        str(len(df)), 
        tp, 
        tn, 
        fp, 
        fn, 
        (tn+tp) / len(df), 
        tp / (tp+fp), 
        tp / (tp+fn), 
        SEMANTIC_THRESHOLD, 
        f1
        ]

    actual = pd.Series(df["is_paraphrase"], name='Actual')
    predicted = pd.Series(df[FUZZY_BIN], name='Predicted')
    df_cross = pd.crosstab(actual, predicted)
    f1 = f1_score(actual, predicted)

    eval_df.loc[len(eval_df.index)] = [
        file.split("_")[0], 
        "fuzzy", 
        str(len(df)), 
        tp, 
        tn, 
        fp, 
        fn, 
        (tn+tp) / len(df), 
        tp / (tp+fp), 
        tp / (tp+fn), 
        FUZZY_THRESHOLD, 
        f1
        ]


eval_df.to_json(os.path.join(OUT_DIR, EVALUATION_FOLDER, EVALUATION_RESULTS_FILENAME), orient = "index", index = True, indent = 4)
eval_df.to_csv(os.path.join(OUT_DIR, EVALUATION_FOLDER, EVALUATION_RESULTS_FILENAME.replace(".json", ".csv")), index = True)

print("Done.")
