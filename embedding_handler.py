import os
import pandas as pd
from tqdm import tqdm
from re import sub
import numpy as np
from thefuzz import fuzz
import shortuuid
import xml.etree.ElementTree as ET
import re
import sys
from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex, SoftCosineSimilarity, Similarity
from setup import *
from matplotlib import cm
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from numpy import reshape
import seaborn as sns
import pandas as pd  

df = pd.read_json(os.path.join(OUT_DIR, FORMATTED_DATA_FILENAME), orient = "index")
df[TEXTEMBED1] = None
df[TEXTEMBED2] = None
df[COSINE_DISTANCE] = None



device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True ).to(device)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

print("GPU available: " +  str(torch.cuda.is_available()))
print("Model running on " + device)

#Check for paraphrase with fuzzy based
t1_embeddings = []
t2_embeddings = []
print("Creating embeddings for each sentence (text1 & text2) and calculating their distances ...")
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    if i>100:
        break

    # mark the text with BERT special characters
    t_1 = "[CLS] " + row[TEXT1].replace(".", ". [SEP][CLS]") 
    t_2 = "[CLS] " + row[TEXT2].replace(".", ". [SEP][CLS]") 
    if t_1.endswith("[CLS]"):
        t_1 = t_1[:-5]
    if not t_1.endswith("[SEP]"):
        t_1 = t_1 + " [SEP]"
    if t_2.endswith("[CLS]"):
        t_2 = t_2[:-5]
    if not t_2.endswith("[SEP]"):
        t_2 = t_2 + " [SEP]"

    # tokenize with BERT tokenizer
    t1_tokenized = tokenizer.tokenize(t_1)
    t2_tokenized = tokenizer.tokenize(t_2)

    # throw out longer that 512 token texts because BERT model struggels to process them
    if len(t1_tokenized) > 512 or len(t2_tokenized) > 512:
        continue

    # map tokens to vocab indices
    t1_indexed = tokenizer.convert_tokens_to_ids(t1_tokenized)
    t2_indexed = tokenizer.convert_tokens_to_ids(t2_tokenized)

    t1_segments_ids = [1] * len(t1_tokenized)
    t2_segments_ids = [1] * len(t2_tokenized)

    #Extract Embeddings
    t1_tensor = torch.tensor([t1_indexed])
    t1_segments_tensors = torch.tensor([t1_segments_ids])
    t2_tensor = torch.tensor([t2_indexed])
    t2_segments_tensors = torch.tensor([t2_segments_ids])

    # collect all of the hidden states produced from all layers 
    with torch.no_grad():
        t1_hidden_states = model(t1_tensor, t1_segments_tensors)[2]
        t2_hidden_states = model(t2_tensor, t2_segments_tensors)[2]

    # Concatenate the tensors for all layers (create a new dimension in the tensor)
    t1_embeds = torch.stack(t1_hidden_states, dim=0)
    t2_embeds = torch.stack(t2_hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    t1_embeds = torch.squeeze(t1_embeds, dim=1)
    t2_embeds = torch.squeeze(t2_embeds, dim=1)

    #Switch dimensions
    t1_embeds = t1_embeds.permute(1,0,2)
    t2_embeds = t2_embeds.permute(1,0,2)

    # Create Word Vector Representation for all tokens within the sentences
    t1_token_vecs = []
    t2_token_vecs = []
    for token in t1_embeds:
        # Concatenate the vectors (that is, append them together) from the last four layers.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        t1_token_vecs.append(cat_vec)
    for token in t2_embeds:
        # Concatenate the vectors (that is, append them together) from the last four layers.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        t2_token_vecs.append(cat_vec)

    # Create Sentence Vector Representations (average of all token vectors)
    text1_embedding = torch.mean(t1_hidden_states[-2][0], dim=0)
    text2_embedding = torch.mean(t2_hidden_states[-2][0], dim=0)

    t1_embeddings.append(np.array(list(text1_embedding)))
    t2_embeddings.append(np.array(list(text2_embedding)))

    cos_distance = 1 - cosine(text1_embedding, text2_embedding)

    #df.at[i, TEXTEMBED1] = text1_embedding
    #df.at[i, TEXTEMBED2] = text2_embedding
    df.at[i, COSINE_DISTANCE] = cos_distance


model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=23)
np.set_printoptions(suppress=True)
tsne = model.fit_transform(t1_embeddings)
coord_x = tsne[:, 0]
coord_y = tsne[:, 1]

print(coord_x)
print(coord_y)

labels = df[TEXT1].tolist()
# Plot sentences
plt.figure(figsize=(20, 20))
plt.scatter(coord_x, coord_y, s=100,alpha=.5)

for j in range(len(t1_embeddings)):
    plt.annotate(
        labels[j][:19],
        xy=(coord_x[j], coord_y[j]),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right', va='bottom')

plt.show()


# Output data
df.to_json(os.path.join(OUT_DIR, "data_embedded.json"), orient = "index", index = True, indent = 4)

torch.save(model, os.path.join(OUT_DIR, "embeddings_model"))