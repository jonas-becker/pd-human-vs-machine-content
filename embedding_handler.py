import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys
from setup import *
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import pandas as pd  
import plotly.express as px
import gc

df = pd.read_json(os.path.join(OUT_DIR, FORMATTED_DATA_FILENAME), orient = "index")
df[COSINE_DISTANCE] = None
stats_string = "EMBEDDING STATISTICS \n\nThese are te statistics gathered during the embedding process. \n\n"

def create_tsne_figure(embed_data, out_filename):
    dataset_embeddings = [d[EMBED] for d in embed_data]
    text_ids = [d[TEXT_ID] for d in embed_data]
    pair_ids = [d[PAIR_ID] for d in embed_data]
    pair_paraphrased = [d[PARAPHRASE] for d in embed_data]
    text_previews = [d[TEXT_PREVIEW] for d in embed_data]
    tuple_markers = [d[TUPLE_ID] for d in embed_data]   #false= first text, true= second text
    tsne_model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=23)
    np.set_printoptions(suppress=True)
    tsne = tsne_model.fit_transform(dataset_embeddings)
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
        hover_data=["text_id", "pair_id", "text", "paraphrase"], title = 'Embedding Visualization: ' + out_filename
        )
    fig.update_layout(showlegend=False)
    #fig.show()
    fig.write_html(os.path.join(OUT_DIR, FIGURES_FOLDER, out_filename + ".html"))

def visualize_embeddings(embed_dict, dataset, stats_string):
    # Create visualization per dataset (paraphrase & non-paraphrase pairs combined)
    print("Creating visualizations per dataset...")
    stats_string = stats_string + "\nTotal Data:\n"
    embed_data = embed_dict[dataset][EMBEDDINGS]
    print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + "dataset. That makes " + str(len(embed_data)/2) + " text pairs.")
    stats_string = stats_string + dataset + ": " + str(len(embed_data)) + " texts, "+ str(int(len(embed_data)/2)) + "pairs \n"
    create_tsne_figure(embed_data, dataset+"_embeddings")

    # Create visualization per dataset (paraphrase pairs only)
    print("Creating visualizations per dataset (paraphrase-pairs only)...")
    stats_string = stats_string + "\nParaphrased Pairs Only Data:\n"
    embed_data = embed_dict[dataset][EMBEDDINGS]
    # Filter the data to only contain paraphrased pairs:
    embed_data = [e for e in embed_data if e[PARAPHRASE] == True]
    print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + "dataset. That makes " + str(len(embed_data)/2) + " text pairs.")
    stats_string = stats_string + dataset + ": " + str(len(embed_data)) + " texts, "+ str(int(len(embed_data)/2)) + "pairs \n"
    create_tsne_figure(embed_data, dataset+"_paraphrasedOnly_embeddings")

    '''
    # Create visualization Machine-Paraphrased pairs
    if dataset in MACHINE_PARAPHRASED_DATASETS:
        print("Creating visualizations per dataset (machine paraphrase-pairs)...")
        stats_string = stats_string + "\nMachine-Pairs Data:\n"
        local_embed_data = embed_dict[dataset][EMBEDDINGS]
        # Filter the data to only contain paraphrased pairs:
        local_embed_data = [e for e in local_embed_data if e[PARAPHRASE] == True]
        embed_data = embed_data + local_embed_data
        stats_string = stats_string + str(len(embed_data)) + " texts, "+ str(int(len(embed_data)/2)) + "pairs \n"
        create_tsne_figure(embed_data, "machine_embeddings")

    # Create visualization Human-Paraphrased pairs
    if dataset not in MACHINE_PARAPHRASED_DATASETS:
        print("Creating visualizations per dataset (human paraphrase-pairs)...")
        stats_string = stats_string + "\nHuman-Pairs Data:\n"
        local_embed_data = embed_dict[dataset][EMBEDDINGS]
        # Filter the data to only contain paraphrased pairs:
        local_embed_data = [e for e in local_embed_data if e[PARAPHRASE] == True]
        embed_data = embed_data + local_embed_data
        stats_string = stats_string + str(len(embed_data)) + " texts, "+ str(int(len(embed_data)/2)) + "pairs \n\n"
        create_tsne_figure(embed_data, "human_embeddings")
    '''
    return stats_string

def calculate_cosine_dists(df, embed_dict, dataset, stats_string):
    # Calculate cosine distance of embeddings between each pair
    print("Calculating cosine distances between pairs embeddings...")
    embed_data = embed_dict[dataset][EMBEDDINGS]

    dataset_embeddings = [[d[EMBED], d[TEXT_ID]] for d in embed_data]
    pair_ids = list(set([d[PAIR_ID] for d in embed_data]))

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

    # Calculate the mean distances per dataset (non-paraphrases and paraphrases seperately)
    df_filtered = df[(df[DATASET] == dataset) & (df[PARAPHRASE] == True)]
    mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
    print("The mean cosine distance of paraphrased pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))
    stats_string = stats_string + "Mean cosine distance of paraphrased pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist) + "\n"
    
    df_filtered = df[(df[DATASET] == dataset) & (df[PARAPHRASE] == False)]
    mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
    print("The mean cosine distance of non-paraphrased (original) pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))
    stats_string = stats_string + "Mean cosine distance of non-paraphrased pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist) + "\n\n"

    return df, stats_string

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True ).to(device)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

print("GPU available: " +  str(torch.cuda.is_available()))
print("Model running on " + device)

#Check for paraphrase with fuzzy based
embed_dict = { }
tokenized_texts = { }
tokenized_pairs = { }

print("Creating embeddings for each sentence (text1 & text2) ...")
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
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

    tokenized_pairs[row[PAIR_ID]] = {ID1: row[ID1], TOKENS1: t1_tokenized, ID2: row[ID2], TOKENS2: t2_tokenized, PARAPHRASE: row[PARAPHRASE], TEXT_PREVIEW1: row[TEXT1][:90]+"...", TEXT_PREVIEW2: row[TEXT2][:90]+"...", DATASET: row[DATASET]}

last_dataset_viewed = tokenized_pairs[list(tokenized_pairs.keys())[0]][DATASET]

for i, pair_id in enumerate(tqdm(list(tokenized_pairs.keys()))):

    dataset = tokenized_pairs[pair_id][DATASET]
    if dataset != last_dataset_viewed or i == len(tokenized_pairs)-1:
        stats_string = visualize_embeddings(embed_dict, last_dataset_viewed, stats_string)
        df, stats_string = calculate_cosine_dists(df, embed_dict, last_dataset_viewed, stats_string)
        
        df[df[DATASET] == last_dataset_viewed].to_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, last_dataset_viewed+"_embedded.json"), orient = "index", index = True, indent = 4)
        last_dataset_viewed = dataset
        embed_dict = { }
        gc.collect()

    # throw out longer that 512 token texts because BERT model struggels to process them
    if len(tokenized_pairs[pair_id][TOKENS1]) > 512 or len(tokenized_pairs[pair_id][TOKENS2]) > 512:
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

    # Add embeddings to dataset-specific lists:
    if dataset in embed_dict:
        embed_dict[dataset][EMBEDDINGS].append({ PAIR_ID: pair_id, TEXT_ID: tokenized_pairs[pair_id][ID1], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW1], EMBED: list(embedding_1), PARAPHRASE: tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: False }) 
        embed_dict[dataset][EMBEDDINGS].append({ PAIR_ID: pair_id, TEXT_ID: tokenized_pairs[pair_id][ID2], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW2], EMBED: list(embedding_2), PARAPHRASE: tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: True }) 
    else:
        embed_dict[dataset] = { EMBEDDINGS: [{ PAIR_ID: pair_id, TEXT_ID: tokenized_pairs[pair_id][ID1], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW1], EMBED: list(embedding_1), PARAPHRASE: tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: False }] }
        embed_dict[dataset][EMBEDDINGS].append({ PAIR_ID: pair_id, TEXT_ID: tokenized_pairs[pair_id][ID2], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW2], EMBED: list(embedding_2), PARAPHRASE: tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: True }) 

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
'''

# Create visualization per dataset (paraphrase & non-paraphrase pairs combined)
print("Creating visualizations per dataset...")
stats_string = stats_string + "\nTotal Data:\n"
for dataset in tqdm(list(embed_dict.keys())):
    embed_data = embed_dict[dataset][EMBEDDINGS]
    print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + "dataset. That makes " + str(len(embed_data)/2) + " text pairs.")
    stats_string = stats_string + dataset + ": " + str(len(embed_data)) + " texts, "+ str(int(len(embed_data)/2)) + "pairs \n"
    create_tsne_figure(embed_data, dataset+"_embeddings")


# Create visualization per dataset (paraphrase pairs only)
print("Creating visualizations per dataset (paraphrase-pairs only)...")
stats_string = stats_string + "\nParaphrased Pairs Only Data:\n"
for dataset in tqdm(list(embed_dict.keys())):
    embed_data = embed_dict[dataset][EMBEDDINGS]
    # Filter the data to only contain paraphrased pairs:
    embed_data = [e for e in embed_data if e[PARAPHRASE] == True]
    print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + "dataset. That makes " + str(len(embed_data)/2) + " text pairs.")
    stats_string = stats_string + dataset + ": " + str(len(embed_data)) + " texts, "+ str(int(len(embed_data)/2)) + "pairs \n"
    create_tsne_figure(embed_data, dataset+"_paraphrasedOnly_embeddings")

# Create visualization Machine-Paraphrased pairs
print("Creating visualizations per dataset (machine paraphrase-pairs)...")
stats_string = stats_string + "\nMachine-Pairs Data:\n"
embed_data = []
for dataset in tqdm(list(embed_dict.keys())):
    print(dataset)
    if dataset not in MACHINE_PARAPHRASED_DATASETS:
        continue
    local_embed_data = embed_dict[dataset][EMBEDDINGS]
    # Filter the data to only contain paraphrased pairs:
    local_embed_data = [e for e in local_embed_data if e[PARAPHRASE] == True]
    embed_data = embed_data + local_embed_data
stats_string = stats_string + str(len(embed_data)) + " texts, "+ str(int(len(embed_data)/2)) + "pairs \n"
create_tsne_figure(embed_data, "machine_embeddings")

# Create visualization Human-Paraphrased pairs
print("Creating visualizations per dataset (human paraphrase-pairs)...")
stats_string = stats_string + "\nHuman-Pairs Data:\n"
embed_data = []
for dataset in tqdm(list(embed_dict.keys())):
    print(dataset)
    if dataset in MACHINE_PARAPHRASED_DATASETS:
        continue
    local_embed_data = embed_dict[dataset][EMBEDDINGS]
    # Filter the data to only contain paraphrased pairs:
    local_embed_data = [e for e in local_embed_data if e[PARAPHRASE] == True]
    embed_data = embed_data + local_embed_data
stats_string = stats_string + str(len(embed_data)) + " texts, "+ str(int(len(embed_data)/2)) + "pairs \n\n"
create_tsne_figure(embed_data, "human_embeddings")

# Calculate cosine distance of embeddings between each pair
print("Calculating cosine distances between pairs embeddings...")
for dataset in tqdm(list(embed_dict.keys())):
    embed_data = embed_dict[dataset][EMBEDDINGS]

    dataset_embeddings = [[d[EMBED], d[TEXT_ID]] for d in embed_data]
    pair_ids = list(set([d[PAIR_ID] for d in embed_data]))

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

# Calculate the mean distances per dataset (non-paraphrases and paraphrases seperately)
for dataset in DATASETS:
    df_filtered = df[(df[DATASET] == dataset) & (df[PARAPHRASE] == True)]
    mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
    print("The mean cosine distance of paraphrased pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))
    stats_string = stats_string + "Mean cosine distance of paraphrased pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist) + "\n"
    
    df_filtered = df[(df[DATASET] == dataset) & (df[PARAPHRASE] == False)]
    mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
    print("The mean cosine distance of non-paraphrased (original) pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))
    stats_string = stats_string + "Mean cosine distance of non-paraphrased pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist) + "\n\n"

# Output data
df.to_json(os.path.join(OUT_DIR, "data_embedded.json"), orient = "index", index = True, indent = 4)
'''

# Output Stats
with open(os.path.join(OUT_DIR, "stats_embedding.txt"), encoding="utf8", mode = "w") as f:
    f.write(stats_string)