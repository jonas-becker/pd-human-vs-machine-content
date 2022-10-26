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
from sklearn.metrics import PrecisionRecallDisplay
import seaborn as sns
from sklearn.metrics import classification_report

def plot_pr_curve(title, y_pred, y_test):
    display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, name="LinearSVC")
    _ = display.ax_.set_title(title)
    _ = display.ax_.plot()
    _ = display.ax_.set_xlabel('Recall')
    _ = display.ax_.set_ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, title+".pdf"), bbox_inches='tight')

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


c_pal = sns.color_palette("colorblind")
plt.rcParams.update({'font.size': 30})

eval_df = pd.DataFrame(columns=[DATASET_NAME, METHOD, PAIRS, TP, TN, FP, FN, ACCURACY, PRECISION, RECALL, SPECIFICITY, F1])

eval_string = ""
eval_string += "-------------------------" + "\n"
eval_string += "OVERALL DATA STATISTICS" + "\n"
eval_string += "-------------------------" + "\n"
eval_string += "Processed datasets: "
for d in DATASETS:
    eval_string += d + ", "
eval_string = eval_string[:-2] + "\n\n"


# find the optimal thresholds for each method and dataset and evaluate with said thresholds
df_test = pd.read_json(os.path.join(OUT_DIR, DETECTION_FOLDER, "detection_test_result.json"), orient="index")

for dataset in DATASETS:
    df = df_test[df_test[DATASET] == dataset]
    print(f"---> Evaluating {dataset}...")
    print("Data size of the " + str(dataset) + " test split: " + str(len(df)))

    #evaluate classification
    #val_df, eval_string = eval(df, dataset, SEM_BERT, eval_df, eval_string)

    print("Calculating precision-recall curves...")
    plot_pr_curve("BERT_"+dataset+"_pr", df[SEM_BERT].tolist(), df[PARAPHRASE].tolist())
    #precision_sem_bert, recall_sem_bert, thresholds_sem_bert = precision_recall_curve(df[PARAPHRASE], df[SEM_BERT])
    #precision_sem_t5, recall_sem_t5, thresholds_sem_t5 = precision_recall_curve(df["is_paraphrase"], df[SEM_T5])
    #precision_fuzzy, recall_fuzzy, thresholds_fuzzy = precision_recall_curve(df["is_paraphrase"], df[FUZZY])
    #precision_tfidf, recall_tfidf, thresholds_tfidf = precision_recall_curve(df["is_paraphrase"], df[TFIDF_COSINE])
    #precision_ngram, recall_ngram, thresholds_ngram = precision_recall_curve(df["is_paraphrase"], df[NGRAM3])
    #df_gpt3 = df[df[SEM_GPT3].notnull()]
    #if df_gpt3.shape[0] != 0:
    #    precision_gpt3, recall_gpt3, thresholds_gpt3 = precision_recall_curve(df_gpt3["is_paraphrase"], df_gpt3[SEM_GPT3])

    print("Plotting curves...")
    #fig, ax = plt.subplots()
    #ax.plot(recall_sem_bert, precision_sem_bert, color=c_pal[0], label='BERT')
    #ax.plot(recall_sem_t5, precision_sem_t5, color=c_pal[1], label='T5')
    #ax.plot(recall_fuzzy, precision_fuzzy, color=c_pal[2], label='Fuzzy')
    #ax.plot(recall_tfidf, precision_tfidf, color=c_pal[3], label='TF-IDF')
    #ax.plot(recall_ngram, precision_ngram, color=c_pal[4], label='3-Gram')
    #if df_gpt3.shape[0] != 0:
    #    ax.plot(recall_gpt3, precision_gpt3, color=c_pal[5], label='GPT-3')

    #ax.set_xlabel('Recall')
    #ax.set_ylabel('Precision')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    #plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, dataset+"_pr.pdf"), bbox_inches='tight')

#if df_gpt3.shape[0] != 0:
#    legend = plt.legend(loc = 3, framealpha=1, ncol=len(ax.lines))
#    for line in legend.get_lines():
#        line.set_linewidth(3.0)
#    legend_fig = legend.figure
#    legend_fig.canvas.draw()
#    bbox  = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
#    legend_fig.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, "legend.pdf"), dpi="figure", bbox_inches=bbox)


eval_df.to_json(os.path.join(OUT_DIR, EVALUATION_FOLDER, EVALUATION_RESULTS_FILENAME), orient = "index", index = True, indent = 4)
eval_df.to_csv(os.path.join(OUT_DIR, EVALUATION_FOLDER, EVALUATION_RESULTS_FILENAME.replace(".json", ".csv")), index = True)
with open(os.path.join(OUT_DIR, EVALUATION_FOLDER, "eval_stats.txt"), 'w') as f:
    f.write(eval_string)

# Make correlation graphs (mixed)
plt.rcParams.update({'font.size': 16})
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
np.fill_diagonal(mask, False)
sns.heatmap(corr_h_matrix, mask=mask, annot=True, xticklabels=["TF-IDF", "3-Gram", "Fuzzy", "GPT-3", "BERT", "T5"], yticklabels=["TF-IDF", "3-Gram", "Fuzzy", "GPT-3", "BERT", "T5"], vmin=-1, vmax=1)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_human.pdf"), bbox_inches="tight")

plt.clf()
corr_m_matrix = corr_df_m.drop(columns=[PARAPHRASE, COSINE_DISTANCE]).corr()
sns.heatmap(corr_m_matrix, mask=mask, annot=True, xticklabels=["TF-IDF", "3-Gram", "Fuzzy", "GPT-3", "BERT", "T5"], yticklabels=["TF-IDF", "3-Gram", "Fuzzy", "GPT-3", "BERT", "T5"], vmin=-1, vmax=1)   #, xticklabels=None, yticklabels=None
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_machine.pdf"), bbox_inches="tight")
plt.clf()

# Generate Correlation Matrix & Output Heatmap (without NGram)
plt.clf()
corr_h_matrix = corr_df_h.drop(columns=[PARAPHRASE, COSINE_DISTANCE, NGRAM3]).corr()
mask = np.triu(np.ones_like(corr_h_matrix, dtype=bool)) # Generate a mask for the upper triangle
np.fill_diagonal(mask, False)
sns.heatmap(corr_h_matrix, mask=mask, annot=True, xticklabels=["TF-IDF", "Fuzzy", "GPT-3", "BERT", "T5"], yticklabels=["TF-IDF", "Fuzzy", "GPT-3", "BERT", "T5"], vmin=-1, vmax=1)
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_human_nongram.pdf"), bbox_inches="tight")

plt.clf()
corr_m_matrix = corr_df_m.drop(columns=[PARAPHRASE, COSINE_DISTANCE, NGRAM3]).corr()
sns.heatmap(corr_m_matrix, mask=mask, annot=True, xticklabels=["TF-IDF", "Fuzzy", "GPT-3", "BERT", "T5"], yticklabels=["TF-IDF", "Fuzzy", "GPT-3", "BERT", "T5"], vmin=-1, vmax=1)   #, xticklabels=None, yticklabels=None
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUT_DIR, EVALUATION_FOLDER, CORRELATIONS_FOLDER, "corr_machine_nongram.pdf"), bbox_inches="tight")
plt.clf()

print("Done.")
