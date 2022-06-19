import os
import xml.etree.ElementTree as ET

DATASETS_FOLDER = "datasets"    #the folder that contains the dataset directories to read in
FORMATTED_DATA_FILENAME = "true_data.json"  #the name of the file that contains the data to read in
DATASETS = ["MPC", "ETPC", "SAv2", "TURL"]     #the folders in the DATASETS_FOLDER should be named like the datasets here
MACHINE_PARAPHRASED_DATASETS = ["MPC", "SAv2"]
OUT_DIR = "output"      #the directory to output the formatted json in
FIGURES_FOLDER = "figures"

FUZZY = "fuzzy_based_result"
SEMANTIC = "semantic_based_result"

STOPWORDS = ['the', 'and', 'are', 'a']

# Variable Names for the outputs:
TEXT1 = "text_1"
TEXT2 = "text_2"
DATASET = "dataset"
PAIR_ID = "pair_id"
TUPLE_ID = "tuple_id" 
ID1 = "id_1"
ID2 = "id_2"
PARAPHRASE = "is_paraphrase" 
WORDVECS1 = "word_vectors_1" 
WORDVECS2 = "word_vectors_2"
TEXTEMBED1 = "text_embedding_1" 
TEXTEMBED2 = "text_embedding_2"
COSINE_DISTANCE = "cosine_distance"

PARAPHRASE_TYPE = "paraphrase_type"
TYPE_ID = "type_id"
TEXT1_SCOPE = "text1_scope"     # the token id scope x,y that marks the part of sentence which has been modified
TEXT2_SCOPE = "text2_scope"
SENSE_PRESERVING = "sense_preserving"


TRAIN_LABELS = [True, False]

# Variables for Embeddings
TOKENS = "tokens"
TOKENS1 = "tokens_1"
TOKENS2 = "tokens_2"
TEXT_PREVIEW = "text_preview"
TEXT_PREVIEW1 = "text_preview_1"
TEXT_PREVIEW2 = "text_preview_2"
EMBEDDINGS = "embeddings"
TEXT_ID = "text_id"
EMBED = "embed"

# Paraphrase Types (EPT Annotation)
PARAPHRASE_TYPES = {}
TYPE_NAME = "type_name"
TYPE_CATEGORY = "type_category"

with open(os.path.join(os.path.join(DATASETS_FOLDER, "ETPC"), "paraphrase_types.xml"), encoding='utf-8', mode = "r") as file:
    tree = ET.parse(file)
    root = tree.getroot()
    for i, elem in enumerate(root):
        PARAPHRASE_TYPES[int(elem[0].text)] = { TYPE_NAME: elem[1].text, TYPE_CATEGORY: elem[2].text }
    PARAPHRASE_TYPES[0] = { TYPE_NAME: "Unknown", TYPE_CATEGORY: "Unknown" }  # Add the "Unknown" type (needed for other unclassified datasets)
