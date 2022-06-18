DATASETS_FOLDER = "datasets"    #the folder that contains the dataset directories to read in
FORMATTED_DATA_FILENAME = "true_data.json"  #the name of the file that contains the data to read in
DATASETS = ["MPC", "ETPC", "SAv2", "TURL"]     #the folders in the DATASETS_FOLDER should be named like the datasets here
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

TRAIN_LABELS = [True, False]

# Variables for Embeddings
TOKENS = "tokens"
TOKENS1 = "tokens_1"
TOKENS2 = "tokens_2"
TEXT_PREVIEW = "text_preview"
TEXT_PREVIEW1 = "text_preview_1"
TEXT_PREVIEW2 = "text_preview_2"
EMBEDDINGS = "embeddings"