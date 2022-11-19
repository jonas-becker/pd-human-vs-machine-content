import os
import xml.etree.ElementTree as ET

DATASETS_FOLDER = "datasets"    #the folder that contains the dataset directories to read in
FORMATTED_DATA_FILENAME = "true_data.json"  #the name of the file that contains the data to read in
EVALUATION_RESULTS_FILENAME = "evaluation.json"
DATASETS = ["ETPC", "SAv2", "TURL", "MPC", "QQP", "ParaNMT", "APH", "APT", "PAWSWiki", "ParaSCI", "MSCOCO", "SaR"]     #the folders in the DATASETS_FOLDER should be named like the datasets here
MACHINE_PARAPHRASED_DATASETS = ["SAv2", "MPC", "ParaNMT", "APT", "PAWSWiki"]
HUMAN_PARAPHRASED_DATASETS = ["ETPC", "TURL", "QQP", "APH", "ParaSCI", "MSCOCO", "SaR"]

OUT_DIR = "output"      #the directory to output the formatted files in
FIGURES_FOLDER = "figures"
EMBEDDINGS_FOLDER = "embeddings"
EXAMPLES_FOLDER = "examples"
DETECTION_FOLDER = "detection"
EVALUATION_FOLDER = "evaluation"
CORRELATIONS_FOLDER = "correlations"
MODELS_FOLDER = "models"

# Methods:
FUZZY = "fuzzy"
SEM_GLOVE = "semantic_glove"
SEM_BERT = "semantic_bert"
SEM_T5 = "semantic_t5"
SEM_GPT3 = "semantic_gpt3"
NGRAM3 = "3gram"
NGRAM4 = "4gram"
NGRAM5 = "5gram"
FASTTEXT = "fasttext"
TFIDF_COSINE = "tfidf_cosine"
DETECTION_METHODS = [FUZZY, NGRAM3, SEM_BERT, SEM_T5, TFIDF_COSINE, SEM_GPT3, FASTTEXT, SEM_GLOVE]

FIGURE_SIZE = 2000  
MAX_DATASET_INPUT = 10000   # how many pairs to parse per dataset (sampled randomly from bigger datasets)
EXAMPLE_AMOUNT = 500    # how many examples to extract (top sim, low sim & random sim)
STUDY_EXAMPLE_AMOUNT = 10   # amount of extracted examples for human study

# Variable Names for the outputs:
TEXT1 = "text_1"
TEXT2 = "text_2"
DATASET = "dataset"
PAIR_ID = "pair_id"
SPLIT = "split_original"     # should be either "test", "train", "dev", "val" or None, NOT the splits from this study, but from the original datasets if provided
TUPLE_ID = "tuple_id" 
ID1 = "id_1"
ID2 = "id_2"
PARAPHRASE = "is_paraphrase"
COSINE_DISTANCE = "cosine_distance"
ORIGIN = "origin"
SUPPLEMENT_FROM = "supplement_from"
PARAPHRASE_TYPE = "paraphrase_type"
# ETPC-relevant attributes:
TYPE_ID = "type_id"     # the ETP annotation for the paraphrase type
TEXT1_SCOPE = "text1_scope"     # the token id scope x,y that marks the part of sentence which has been modified
TEXT2_SCOPE = "text2_scope"
SENSE_PRESERVING = "sense_preserving"

TRAIN_SPLIT_MAX = 4000      # max amount of train split pairs used for grid search SVM training

# Variables for embeddings and their visualization
TOKENS1 = "tokens_1"
TOKENS2 = "tokens_2"
TEXT_PREVIEW = "text_preview"
TEXT_PREVIEW1 = "text_preview_1"
TEXT_PREVIEW2 = "text_preview_2"
EMBEDDINGS = "embeddings"
TEXT_ID = "text_id"
EMBED = "embed"

# Eval Variables
DATASET_COLUMN_NAME = "dataset"
METHOD = "detection_method"
PAIRS = "pairs"
TP = "TP"
TN = "TN"
FP = "FP"
FN = "FN"
ACCURACY = "accuracy"
PRECISION = "precision"
SPECIFICITY = "specificity"
RECALL = "recall"
F1 = "f1"

CORR_GRAPH_SIZE = 3000

#with open(os.path.join(os.path.join(DATASETS_FOLDER, "ETPC"), "paraphrase_types.xml"), encoding='utf-8', mode = "r") as file:
#    tree = ET.parse(file)
#    root = tree.getroot()
#    for i, elem in enumerate(root):
#        PARAPHRASE_TYPES[int(elem[0].text)] = { TYPE_NAME: elem[1].text, TYPE_CATEGORY: elem[2].text }
#    PARAPHRASE_TYPES[0] = { TYPE_NAME: "Unknown", TYPE_CATEGORY: "Unknown" }  # Add the "Unknown" type (needed for other unclassified datasets)

# Paraphrase Types explained
# (numbers different from ETPC Paper as they seem to have skipped some numbers in EPT dev code)
'''

Morphology-based changes
1 Inflectional changes + / -
2 Modal verb changes +
3 Derivational changes +

Lexicon-based changes
4 Spelling changes +
5 Same polarity substitution (habitual) +
6 Same polarity substitution (contextual) + / -
7 Same polarity sub. (named entity) + / -
8 Change of format +

Lexico-syntactic based changes
9 Opposite polarity sub. (habitual) + / -
10 Opposite polarity sub. (contextual) + / -
11 Synthetic/analytic substitution +
13 Converse substitution + / -

Syntax-based changes
14 Diathesis alternation + / -
15 Negation switching + / -
16 Ellipsis +
17 Coordination changes +
18 Subordination and nesting changes +

Discourse-based changes
21 Punctuation changes +
22 Direct/indirect style alternations + / -
23 Sentence modality changes +
24 Syntax/discourse structure changes +

Other changes
25 Addition/Deletion + / -
26 Change of order +
28 Semantic (General Inferences) + / -

Extremes
29 Identity +
30 Non-Paraphrase -
31 Entailment 
32 Synthetic/analytic substitution (named ent.)

100 Negation (independed from paraphrase annotation)

'''
