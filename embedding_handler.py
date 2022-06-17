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
import plotly.express as px

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
embeddings = []
tokenized_texts = { }

print("Creating embeddings for each sentence (text1 & text2) ...")
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    if i>999:
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

    tokenized_texts[row[ID1]] = {"tokens": t1_tokenized, PARAPHRASE: False, "text_preview": row[TEXT1][:40]}
    tokenized_texts[row[ID2]] = {"tokens": t2_tokenized, PARAPHRASE: row[PARAPHRASE], "text_preview": row[TEXT2][:40]}


for tokenized in tqdm(list(tokenized_texts.keys())):
    # throw out longer that 512 token texts because BERT model struggels to process them
    if len(tokenized_texts[tokenized]["tokens"]) > 512:
        print("Thrown element " + str(tokenized) + " out of the list because it contained to much text ( > 512 chars ).")
        del tokenized_texts[tokenized]
        continue

    # map tokens to vocab indices
    indexed = tokenizer.convert_tokens_to_ids(tokenized_texts[tokenized]["tokens"])

    segments_ids = [1] * len(tokenized_texts[tokenized]["tokens"])

    #Extract Embeddings
    tensor = torch.tensor([indexed])
    segments_tensors = torch.tensor([segments_ids])

    # collect all of the hidden states produced from all layers 
    with torch.no_grad():
        hidden_states = model(tensor, segments_tensors)[2]

    # Concatenate the tensors for all layers (create a new dimension in the tensor)
    embeds = torch.stack(hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    embeds = torch.squeeze(embeds, dim=1)

    #Switch dimensions
    embeds = embeds.permute(1,0,2)

    # Create Sentence Vector Representations (average of all token vectors)
    embedding = torch.mean(hidden_states[-2][0], dim=0)

    embeddings.append(np.array(list(embedding)))

    #cos_distance = 1 - cosine(text1_embedding, text2_embedding)

    #df.at[i, TEXTEMBED1] = text1_embedding
    #df.at[i, TEXTEMBED2] = text2_embedding
    #df.at[i, COSINE_DISTANCE] = cos_distance



model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=23)
np.set_printoptions(suppress=True)
tsne = model.fit_transform(embeddings)
coord_x = tsne[:, 0]
coord_y = tsne[:, 1]


# Plot sentences
fig1 = plt.figure(figsize=(20, 20))
ax1 = fig1.add_subplot(111)

plt.scatter(coord_x, coord_y, alpha=.5, color="green", marker='s', label='original')

labels = list(tokenized_texts)
texts = [ tokenized_texts[t]["text_preview"] for t in tokenized_texts ]
for j in range(len(embeddings)):
    plt.annotate(
        labels[j],
        xy=(coord_x[j], coord_y[j]),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right', va='bottom')
    

#plt.show()

print(type(tsne))
df_embeddings = pd.DataFrame(tsne)
print(len(df_embeddings))
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})
df_embeddings = df_embeddings.assign(label= labels)
# We will also add the unmodified base sentences, to make the visualization easier :
df_embeddings = df_embeddings.assign(text= texts)

fig = px.scatter(
    df_embeddings, x='x', y='y',
    color='label', labels={'color': 'label'},
    hover_data=['text'], title = 'Embedding Visualization'
    )
fig.update_layout(showlegend=False)

fig.show()


# Output data
df.to_json(os.path.join(OUT_DIR, "data_embedded.json"), orient = "index", index = True, indent = 4)

torch.save(model, os.path.join(OUT_DIR, "embeddings_model"))