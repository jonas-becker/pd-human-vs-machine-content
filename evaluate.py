from audioop import avg
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
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import seaborn as sns
from sklearn.metrics import classification_report

def plot_pr_curve(title, y_pred, y_test):
    #_, ax = plt.subplots(figsize=(7, 8))

    precision, recall, thresholds = precision_recall_curve(np.array(y_test), np.array(y_pred))

    display = PrecisionRecallDisplay(recall = recall, precision = precision)
    display.plot(name=f"Precision-recall")
    plt.ylim([0.5, 1.0])
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, title+".pdf"), bbox_inches='tight')

def plot_roc_curve(title, y_pred, y_test):
    display = RocCurveDisplay.from_predictions(y_test, y_pred, name="LinearSVC")
    _ = display.ax_.set_title(title)
    _ = display.ax_.plot()
    _ = display.ax_.set_xlabel('FP Rate')
    _ = display.ax_.set_ylabel('TP Rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, title+".pdf"), bbox_inches='tight')

def gini(x):
    '''
    Calculates the Gini Coefficient.
    :param x: The pd series to calculate on
    :return: the Gini Coefficient
    '''
    x = x.reset_index(drop=True)
    w = np.ones_like(x)
    w = pd.Series(w).reset_index(drop=True)
    n = x.size
    wxsum = sum(w * x)
    wsum = sum(w)
    sxw = np.argsort(x)
    sx = x[sxw] * w[sxw]
    sw = w[sxw]
    pxi = np.cumsum(sx) / wxsum
    pci = np.cumsum(sw) / wsum
    g = 0.0
    for i in np.arange(1, n):
        g = g + pxi.iloc[i] * pci.iloc[i - 1] - pci.iloc[i] * pxi.iloc[i - 1]
    return g

def eval(df, dataset, method, eval_df, eval_string):
    actual = pd.Series(df[PARAPHRASE], name='Actual')
    predicted = pd.Series(df[method], name='Predicted')
    df_cross = pd.crosstab(actual, predicted)
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
        dataset,
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
        f1
        ]

    eval_string = eval_string + "------> " + dataset + ":\n" + str(classification_report(actual, predicted)) + "\n\n"

    return eval_df, eval_string

if __name__ == "__main__":

    c_pal = sns.color_palette("colorblind")
    plt.rcParams.update({'font.size': 30})

    eval_df = pd.DataFrame()

    eval_string = ""
    eval_string += "-------------------------" + "\n"
    eval_string += "OVERALL DATA STATISTICS" + "\n"
    eval_string += "-------------------------" + "\n"
    eval_string += "Processed datasets: "
    for d in DATASETS:
        eval_string += d + ", "
    eval_string = eval_string[:-2] + "\n\n"

    df_test = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index")

    for dataset in DATASETS:
        df = df_test[df_test[DATASET] == dataset]
        if df.shape[0] == 0:
            continue

        print(f"---> Evaluating {dataset}...")
        print("Data size of the " + str(dataset) + " test split: " + str(len(df)))

        for method in DETECTION_METHODS:
            print("-> Method: " + method)

            #print("Calculating precision-recall curve...")
            #plot_pr_curve(method + "_" + dataset + "_pr", y_pred=np.array(df[method].tolist()), y_test=np.array(df[PARAPHRASE].tolist()).astype(int))
            #plot_roc_curve(method + "_" + dataset + "_roc", y_pred=np.array(df[method].tolist()), y_test=np.array(df[PARAPHRASE].tolist()).astype(int))

            print("Getting statistics...")
            confusion_mat = confusion_matrix(y_true=df[PARAPHRASE].tolist(), y_pred=df[method+PRED_SUF].tolist())
            fp = confusion_mat[0][1]
            fn = confusion_mat[1][0]
            tp = confusion_mat[1][1]
            tn = confusion_mat[0][0]

            eval_df = pd.concat([eval_df, pd.DataFrame({
                DATASET: dataset,
                METHOD: method,
                PAIRS: df.shape[0],
                PARA_PAIRS: df[df[PARAPHRASE] == True].shape[0],
                ORIG_PAIRS: df[df[PARAPHRASE] == False].shape[0],
                FP: fp,
                FN: fn,
                TP: tp,
                TN: tn,
                F1: f1_score(y_true=df[PARAPHRASE].tolist(), y_pred=df[method+PRED_SUF].tolist()),
                GINI_PRED: gini(df[method+PRED_SUF]),
                GINI_PROB: gini(df[method])
            }, index=[dataset+"_"+method])])

        eval_df = pd.concat([eval_df, pd.DataFrame({
            DATASET: dataset,
            METHOD: None,
            PAIRS: eval_df[eval_df[DATASET] == dataset][PAIRS].mean(),
            PARA_PAIRS: eval_df[eval_df[DATASET] == dataset][PARA_PAIRS].mean(),
            ORIG_PAIRS: eval_df[eval_df[DATASET] == dataset][ORIG_PAIRS].mean(),
            FP: eval_df[eval_df[DATASET] == dataset][FP].mean(),
            FN: eval_df[eval_df[DATASET] == dataset][FN].mean(),
            TP: eval_df[eval_df[DATASET] == dataset][TP].mean(),
            TN: eval_df[eval_df[DATASET] == dataset][TN].mean(),
            F1: eval_df[eval_df[DATASET] == dataset][F1].mean(),
            GINI_PRED: eval_df[eval_df[DATASET] == dataset][GINI_PRED].mean(),
            GINI_PROB: eval_df[eval_df[DATASET] == dataset][GINI_PROB].mean()
        }, index=[dataset + "_MEAN"])])

    # mean per method across datasets
    for method in DETECTION_METHODS:
        eval_df = pd.concat([eval_df, pd.DataFrame({
            DATASET: None,
            METHOD: method,
            PAIRS: eval_df[eval_df[METHOD] == method][PAIRS].mean(),
            PARA_PAIRS: eval_df[eval_df[METHOD] == method][PARA_PAIRS].mean(),
            ORIG_PAIRS: eval_df[eval_df[METHOD] == method][ORIG_PAIRS].mean(),
            FP: eval_df[eval_df[METHOD] == method][FP].mean(),
            FN: eval_df[eval_df[METHOD] == method][FN].mean(),
            TP: eval_df[eval_df[METHOD] == method][TP].mean(),
            TN: eval_df[eval_df[METHOD] == method][TN].mean(),
            F1: eval_df[eval_df[METHOD] == method][F1].mean(),
            GINI_PRED: eval_df[eval_df[METHOD] == method][GINI_PRED].mean(),
            GINI_PROB: eval_df[eval_df[METHOD] == method][GINI_PROB].mean()
        }, index=["MEAN_"+method])])

    eval_df.to_json(os.path.join(OUT_DIR, EVALUATION_FOLDER, EVALUATION_RESULTS_FILENAME), orient = "index", index = True, indent = 4)
    eval_df.to_csv(os.path.join(OUT_DIR, EVALUATION_FOLDER, EVALUATION_RESULTS_FILENAME.replace(".json", ".csv")), index = True)
    with open(os.path.join(OUT_DIR, EVALUATION_FOLDER, "eval_stats.txt"), 'w') as f:
        f.write(eval_string)

    # ------ Correlations

    print("\nGenerating correlation graphs for datasets (mixed).")
    plt.rcParams.update({'font.size': 15})
    # machine-paraphrases & human-paraphrases
    corr_df = shuffle(df_test)\
        .reset_index(drop=True).drop(columns=df_test.columns.difference(DETECTION_METHODS))
    # machine-paraphrases
    corr_df_m = shuffle(df_test[(df_test[DATASET].isin(MACHINE_PARAPHRASED_DATASETS))])\
        .reset_index(drop=True).drop(columns=df_test.columns.difference(DETECTION_METHODS))
    # human-paraphrases
    corr_df_h = shuffle(df_test[(df_test[DATASET].isin(HUMAN_PARAPHRASED_DATASETS))])\
        .reset_index(drop=True).drop(columns=df_test.columns.difference(DETECTION_METHODS))

    for method1 in tqdm(DETECTION_METHODS):
        for method2 in DETECTION_METHODS:
            if method1 != method2:
                # get the correct axis names in nicer writing
                if method1 == SEM_BERT:
                    xlabel = "BERT"
                elif method1 == SEM_T5:
                    xlabel = "T5"
                elif method1 == FUZZY:
                    xlabel = "Fuzzy"
                elif method1 == TFIDF_COSINE:
                    xlabel = "Tf-idf Cosine"
                elif method1 == NGRAM3:
                    xlabel = "3-Gram"
                elif method1 == FASTTEXT:
                    xlabel = "Fasttext"
                elif method1 == SEM_GLOVE:
                    xlabel = "GloVe"
                else:
                    xlabel = "Unknown Method"
                if method2 == SEM_BERT:
                    ylabel = "BERT"
                elif method2 == SEM_T5:
                    ylabel = "T5"
                elif method2 == FUZZY:
                    ylabel = "Fuzzy"
                elif method2 == TFIDF_COSINE:
                    ylabel = "Tf-idf Cosine"
                elif method2 == NGRAM3:
                    ylabel = "3-Gram"
                elif method2 == SEM_GLOVE:
                    ylabel = "GloVe"
                elif method2 == FASTTEXT:
                    ylabel = "Fasttext"
                else:
                    ylabel = "Unknown Method"

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

    print("\nGenerating correlation graphs for datasets (paraphrases only).")
    # machine-paraphrases & human-paraphrases
    corr_df = shuffle(df_test[df_test[PARAPHRASE] == True])\
        .reset_index(drop=True).drop(columns=df_test.columns.difference(DETECTION_METHODS))
    # machine-paraphrases
    corr_df_m = shuffle(df_test[(df_test[PARAPHRASE] == True) & (df_test[DATASET].isin(MACHINE_PARAPHRASED_DATASETS))])\
        .reset_index(drop=True).drop(columns=df_test.columns.difference(DETECTION_METHODS))
    # human-paraphrases
    corr_df_h = shuffle(df_test[(df_test[PARAPHRASE] == True) & (df_test[DATASET].isin(HUMAN_PARAPHRASED_DATASETS))])\
        .reset_index(drop=True).drop(columns=df_test.columns.difference(DETECTION_METHODS))

    for method1 in tqdm(DETECTION_METHODS):
        for method2 in DETECTION_METHODS:
            if method1 != method2:
                if method1 == SEM_BERT:
                    xlabel = "BERT"
                elif method1 == SEM_T5:
                    xlabel = "T5"
                elif method1 == FUZZY:
                    xlabel = "Fuzzy"
                elif method1 == TFIDF_COSINE:
                    xlabel = "Tf-idf Cosine"
                elif method1 == NGRAM3:
                    xlabel = "3-Gram"
                elif method1 == FASTTEXT:
                    xlabel = "Fasttext"
                elif method1 == SEM_GLOVE:
                    xlabel = "GloVe"
                else:
                    xlabel = "Unknown Method"
                if method2 == SEM_BERT:
                    ylabel = "BERT"
                elif method2 == SEM_T5:
                    ylabel = "T5"
                elif method2 == FUZZY:
                    ylabel = "Fuzzy"
                elif method2 == TFIDF_COSINE:
                    ylabel = "Tf-idf Cosine"
                elif method2 == NGRAM3:
                    ylabel = "3-Gram"
                elif method2 == SEM_GLOVE:
                    ylabel = "GloVe"
                elif method2 == FASTTEXT:
                    ylabel = "Fasttext"
                else:
                    ylabel = "Unknown Method"

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


    print("\nGenerate Correlation Matrix & Output Heatmap for probabilities...")
    # machine-paraphrases & human-paraphrases
    corr_df = shuffle(df_test.reset_index(drop=True))\
        .drop(columns=df_test.columns.difference(DETECTION_METHODS))
    # machine-paraphrases
    corr_df_m = shuffle(df_test[(df_test[DATASET].isin(MACHINE_PARAPHRASED_DATASETS))]).reset_index(drop=True)\
        .drop(columns=df_test.columns.difference(DETECTION_METHODS))
    # human-paraphrases
    corr_df_h = shuffle(df_test[(df_test[DATASET].isin(HUMAN_PARAPHRASED_DATASETS))]).reset_index(drop=True)\
        .drop(columns=df_test.columns.difference(DETECTION_METHODS))

    plt.clf()
    corr_h_matrix = corr_df_h.corr()
    mask = np.triu(np.ones_like(corr_h_matrix, dtype=bool)) # Generate a mask for the upper triangle
    np.fill_diagonal(mask, False)
    sns.heatmap(corr_h_matrix, mask=mask, annot=True, xticklabels=corr_df_h.columns.tolist(),
                yticklabels=corr_df_h.columns.tolist(), vmin=0, vmax=1)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_human_prob.pdf"), bbox_inches="tight")

    plt.clf()
    corr_m_matrix = corr_df_m.corr()
    sns.heatmap(corr_m_matrix, mask=mask, annot=True, xticklabels=corr_df_m.columns.tolist(),
                yticklabels=corr_df_m.columns.tolist(), vmin=0, vmax=1)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_machine_prob.pdf"), bbox_inches="tight")
    plt.clf()

    print("\nGenerate Correlation Matrix & Output Heatmap for predictions...")
    # machine-paraphrases & human-paraphrases
    corr_df = shuffle(df_test.reset_index(drop=True)) \
        .drop(columns=df_test.columns.difference(METHOD_PRED_CLASSES))
    # machine-paraphrases
    corr_df_m = shuffle(df_test[(df_test[DATASET].isin(MACHINE_PARAPHRASED_DATASETS))]).reset_index(drop=True) \
        .drop(columns=df_test.columns.difference(METHOD_PRED_CLASSES))
    # human-paraphrases
    corr_df_h = shuffle(df_test[(df_test[DATASET].isin(HUMAN_PARAPHRASED_DATASETS))]).reset_index(drop=True) \
        .drop(columns=df_test.columns.difference(METHOD_PRED_CLASSES))

    plt.clf()
    corr_h_matrix = corr_df_h.corr()
    mask = np.triu(np.ones_like(corr_h_matrix, dtype=bool))  # Generate a mask for the upper triangle
    np.fill_diagonal(mask, False)
    sns.heatmap(corr_h_matrix, mask=mask, annot=True, xticklabels=corr_df_h.columns.tolist(),
                yticklabels=corr_df_h.columns.tolist(), vmin=0, vmax=1)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_human_pred.pdf"), bbox_inches="tight")

    plt.clf()
    corr_m_matrix = corr_df_m.corr()
    sns.heatmap(corr_m_matrix, mask=mask, annot=True, xticklabels=corr_df_m.columns.tolist(),
                yticklabels=corr_df_m.columns.tolist(), vmin=0, vmax=1)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_machine_pred.pdf"), bbox_inches="tight")
    plt.clf()

    print("Done.")
