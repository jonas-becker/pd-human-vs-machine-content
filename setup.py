DATASETS_FOLDER = "datasets"    #the folder that contains the dataset directories to read in
FORMATTED_DATA_FILENAME = "true_data.json"  #the name of the file that contains the data to read in
DATASETS = ["MPC", "MSRP", "ETPC"]     #the folders in the DATASETS_FOLDER should be named like the datasets here
OUT_DIR = "output"      #the directory to output the formatted json in

FUZZY = "fuzzy_based_result"
SEMANTIC = "semantic_based_result"

STOPWORDS = ['the', 'and', 'are', 'a']

# Variable Names for the outputs:
TEXT1 = "text_1"
TEXT2 = "text_2"
DATASET = "dataset"
ID1 = "id_1"
ID2 = "id_2"
PARAPHRASE = "is_paraphrase" 
WORDVECS1 = "word_vectors_1" 
WORDVECS2 = "word_vectors_2"
TEXTEMBED1 = "text_embedding_1" 
TEXTEMBED2 = "text_embedding_2"
COSINE_DISTANCE = "cosine_distance"

TRAIN_LABELS = [True, False]