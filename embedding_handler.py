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
embeddings = []
embed_dict = { }
MPC_embeddings = []
ETPC_embeddings = []

tokenized_texts = { }
tokenized_pairs = { }

print("Creating embeddings for each sentence (text1 & text2) ...")
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    #if i>39:
    #    break

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

    #tokenized_texts[row[ID1]] = {PAIR_ID: row[PAIR_ID], TOKENS: t1_tokenized, PARAPHRASE: False, TEXT_PREVIEW: row[TEXT1][:70]+"...", DATASET: row[DATASET]}
    #tokenized_texts[row[ID2]] = {PAIR_ID: row[PAIR_ID], TOKENS: t2_tokenized, PARAPHRASE: row[PARAPHRASE], TEXT_PREVIEW: row[TEXT2][:70]+"...", DATASET: row[DATASET]}

    tokenized_pairs[row[PAIR_ID]] = {ID1: row[ID1], TOKENS1: t1_tokenized, ID2: row[ID2], TOKENS2: t2_tokenized, PARAPHRASE: row[PARAPHRASE], TEXT_PREVIEW1: row[TEXT1][:90]+"...", TEXT_PREVIEW2: row[TEXT2][:90]+"...", DATASET: row[DATASET]}


for pair_id in tqdm(list(tokenized_pairs.keys())):
    # throw out longer that 512 token texts because BERT model struggels to process them
    if len(tokenized_pairs[pair_id][TOKENS1]) > 512 or len(tokenized_pairs[pair_id][TOKENS2]) > 512:
        print("Thrown element " + str(pair_id) + " out of the list because it contained to much text ( > 512 chars ).")
        del tokenized_pairs[pair_id]
        continue
    
    # DO FOR FIRST TEXT
    # map tokens to vocab indices
    indexed = tokenizer.convert_tokens_to_ids(tokenized_pairs[pair_id][TOKENS1])
    segments_ids = [1] * len(tokenized_pairs[pair_id][TOKENS1])
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
    embedding_1 = torch.mean(hidden_states[-2][0], dim=0)
    embeddings.append(np.array(list(embedding_1)))

    # DO FOR SECOND TEXT
    # map tokens to vocab indices
    indexed = tokenizer.convert_tokens_to_ids(tokenized_pairs[pair_id][TOKENS2])
    segments_ids = [1] * len(tokenized_pairs[pair_id][TOKENS2])
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
    embedding_2 = torch.mean(hidden_states[-2][0], dim=0)
    embeddings.append(np.array(list(embedding_2)))

    # Add embeddings to dataset-specific lists:
    dataset = tokenized_pairs[pair_id][DATASET]
    if dataset in embed_dict:
        embed_dict[dataset][EMBEDDINGS].append({ "pair_id": pair_id, "text_id": tokenized_pairs[pair_id][ID1], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW1] ,"embed": list(embedding_1), "paraphrased_pair": tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: False }) 
        embed_dict[dataset][EMBEDDINGS].append({ "pair_id": pair_id, "text_id": tokenized_pairs[pair_id][ID2], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW2] ,"embed": list(embedding_2), "paraphrased_pair": tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: True }) 
    else:
        embed_dict[dataset] = { EMBEDDINGS: [{ "pair_id": pair_id, "text_id": tokenized_pairs[pair_id][ID1], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW1] ,"embed": list(embedding_1), "paraphrased_pair": tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: False }] }
        embed_dict[dataset][EMBEDDINGS].append({ "pair_id": pair_id, "text_id": tokenized_pairs[pair_id][ID2], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW2] ,"embed": list(embedding_2), "paraphrased_pair": tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: True }) 

# Save dataset-specific embeddings
#embed_dict["ETPC"] = { DATASET: "ETPC", EMBEDDINGS: ETPC_embeddings}
#embed_dict["MPC"] = { DATASET: "MPC", EMBEDDINGS: MPC_embeddings}

# Create T-SNE visualization for whole data (all datasets combined)
'''
print("Creating the visualization for all datasets combined...")
tsne_model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=23)
np.set_printoptions(suppress=True)
tsne = tsne_model.fit_transform(embeddings)
coord_x = tsne[:, 0]
coord_y = tsne[:, 1]
labels = list(tokenized_texts)
texts = [ tokenized_texts[t][TEXT_PREVIEW] for t in tokenized_texts ]

df_embeddings = pd.DataFrame(tsne)
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})
df_embeddings = df_embeddings.assign(label= labels)
df_embeddings = df_embeddings.assign(text= texts)
fig = px.scatter(
    df_embeddings, x='x', y='y',
    color='label', labels={'color': 'label'},
    hover_data=['text'], title = 'Embedding Visualization (T-SNE)'
    )
fig.update_layout(showlegend=False)
fig.show()
fig.write_html(os.path.join(OUT_DIR, FIGURES_FOLDER, "all_embeddings.html"))
'''

# Create visualization per dataset (paraphrase & non-paraphrase pairs combined)
print("Creating visualizations per dataset...")
for dataset in tqdm(list(embed_dict.keys())):
    embed_data = embed_dict[dataset][EMBEDDINGS]

    print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + "dataset. That makes " + str(len(embed_data)/2) + " text pairs.")

    dataset_embeddings = [d["embed"] for d in embed_data]
    text_ids = [d["text_id"] for d in embed_data]
    pair_ids = [d["pair_id"] for d in embed_data]
    pair_paraphrased = [d["paraphrased_pair"] for d in embed_data]
    text_previews = [d[TEXT_PREVIEW] for d in embed_data]
    tuple_markers = [d[TUPLE_ID] for d in embed_data]   #false= first text, true= second text

    tsne_model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=23)
    np.set_printoptions(suppress=True)
    tsne = tsne_model.fit_transform(dataset_embeddings)
    coord_x = tsne[:, 0]
    coord_y = tsne[:, 1]

    df_embeddings = pd.DataFrame(tsne)
    df_embeddings = df_embeddings.rename(columns = {0:'x',1:'y'})
    df_embeddings = df_embeddings.assign(label = text_ids)
    df_embeddings = df_embeddings.assign(text_id = text_ids)
    df_embeddings = df_embeddings.assign(paraphrase = pair_paraphrased)
    df_embeddings = df_embeddings.assign(text = text_previews)
    df_embeddings = df_embeddings.assign(tuple_marker= tuple_markers)
    df_embeddings = df_embeddings.assign(pair_id = pair_ids)

    fig = px.scatter(
        df_embeddings, x='x', y='y',
        color='tuple_marker', labels={'color': 'label'},
        hover_data=["text_id", "pair_id", "text", "paraphrase"], title = 'Embedding Visualization: ' + str(dataset)
        )
    fig.update_layout(showlegend=False)
    #fig.show()
    fig.write_html(os.path.join(OUT_DIR, FIGURES_FOLDER, dataset + "_embeddings.html"))


# Create visualization per dataset (paraphrase pairs only)
print("Creating visualizations per dataset (paraphrase-pairs only)...")
for dataset in tqdm(list(embed_dict.keys())):
    embed_data = embed_dict[dataset][EMBEDDINGS]
    # Filter the data to only contain paraphrased pairs:
    embed_data = [e for e in embed_data if e["paraphrased_pair"] == True]

    print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + "dataset. That makes " + str(len(embed_data)/2) + " text pairs.")

    dataset_embeddings = [d["embed"] for d in embed_data]
    text_ids = [d["text_id"] for d in embed_data]
    pair_ids = [d["pair_id"] for d in embed_data]
    pair_paraphrased = [d["paraphrased_pair"] for d in embed_data]
    text_previews = [d[TEXT_PREVIEW] for d in embed_data]
    tuple_markers = [d[TUPLE_ID] for d in embed_data]   #false= first text, true= second text

    tsne_model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=23)
    np.set_printoptions(suppress=True)
    tsne = tsne_model.fit_transform(dataset_embeddings)
    coord_x = tsne[:, 0]
    coord_y = tsne[:, 1]

    df_embeddings = pd.DataFrame(tsne)
    df_embeddings = df_embeddings.rename(columns = {0:'x',1:'y'})
    df_embeddings = df_embeddings.assign(label = text_ids)
    df_embeddings = df_embeddings.assign(text_id = text_ids)
    df_embeddings = df_embeddings.assign(paraphrase = pair_paraphrased)
    df_embeddings = df_embeddings.assign(text = text_previews)
    df_embeddings = df_embeddings.assign(tuple_marker= tuple_markers)
    df_embeddings = df_embeddings.assign(pair_id = pair_ids)

    fig = px.scatter(
        df_embeddings, x='x', y='y',
        color='tuple_marker', labels={'color': 'label'},
        hover_data=["text_id", "pair_id", "text", "paraphrase"], title = 'Embedding Visualization: ' + str(dataset) + ' (paraphrased pairs only)'
        )
    fig.update_layout(showlegend=False)
    #fig.show()
    fig.write_html(os.path.join(OUT_DIR, FIGURES_FOLDER, dataset + "_paraphrasedOnly_embeddings.html"))

# Calculate cosine distance of embeddings between each pair
print("Calculating cosine distances between pairs embeddings...")
for dataset in tqdm(list(embed_dict.keys())):
    embed_data = embed_dict[dataset][EMBEDDINGS]

    dataset_embeddings = [[d["embed"], d["text_id"]] for d in embed_data]
    pair_ids = [d["pair_id"] for d in embed_data]
    pair_ids = list(set(pair_ids))

    for i, pair_id in enumerate(pair_ids):
        id_1 = df.loc[df[PAIR_ID] == pair_id][ID1].item()
        id_2 = df.loc[df[PAIR_ID] == pair_id][ID2].item()
        emb_1 = None
        emb_2 = None
        for d in dataset_embeddings:
            if d[1] == id_1:
                emb_1 = d[0]
            elif d[1] == id_2:
                emb_2 = d[0]
            if emb_1 and emb_2:
                break
        df.loc[df[PAIR_ID] == pair_id, COSINE_DISTANCE] = 1 - cosine(emb_1, emb_2)

# Calculate the mean distances per dataset
for dataset in DATASETS:
    df_filtered = df[(df[DATASET] == dataset) & (df[PARAPHRASE] == True)]
    mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
    print("The mean cosine distance of paraphrased pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))
    
    df_filtered = df[(df[DATASET] == dataset) & (df[PARAPHRASE] == False)]
    mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
    print("The mean cosine distance of non-paraphrased (original) pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))

# Output data
df.to_json(os.path.join(OUT_DIR, "data_embedded.json"), orient = "index", index = True, indent = 4)