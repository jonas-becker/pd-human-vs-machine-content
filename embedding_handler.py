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
from sklearn.utils import shuffle
import pandas as pd  
import plotly.express as px
import json
import gc

def create_tsne_figure(embed_data, out_filename):
    dataset_embeddings = [d[EMBED] for d in embed_data]
    dataset_embeddings = torch.tensor(dataset_embeddings, device = "cpu")   # move tensors to cpu
    text_ids = [d[TEXT_ID] for d in embed_data]
    pair_ids = [d[PAIR_ID] for d in embed_data]
    dataset_ids = [d[DATASET] for d in embed_data]
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
    df_embeddings = df_embeddings.assign(dataset = dataset_ids)
    if "total" in out_filename:
        fig = px.scatter(
            df_embeddings, x='x', y='y',
            color='dataset', labels={'color': 'label'},
            hover_data=["dataset", "text_id", "pair_id", "text", "paraphrase"], title = 'Embedding Visualization: ' + out_filename
            )
        fig.update_layout(showlegend=True)
    else:
        fig = px.scatter(
            df_embeddings, x='x', y='y',
            color='tuple_marker', labels={'color': 'label'},
            hover_data=["text_id", "pair_id", "text", "paraphrase"], title = 'Embedding Visualization: ' + out_filename
            )
        fig.update_layout(showlegend=False)
    #fig.show()
    fig.write_html(os.path.join(OUT_DIR, FIGURES_FOLDER, out_filename + ".html"))
    fig.write_image(os.path.join(OUT_DIR, FIGURES_FOLDER, out_filename + ".pdf"))
    fig.write_image(os.path.join(OUT_DIR, FIGURES_FOLDER, out_filename + ".png"))
    fig.write_image(os.path.join(OUT_DIR, FIGURES_FOLDER, out_filename + ".svg"))

def visualize_embeddings(embed_dict, dataset):

    if dataset == "total":
        # Create visualization for all datasets (paraphrase & non-paraphrase pairs combined)
        print("Creating visualizations for dataset...")
        embed_data = []
        for d in embed_dict.keys():
            embed_data = embed_data + embed_dict[d][EMBEDDINGS]  # combine dataset embeds

        if len(embed_data) > 1:
            print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + " dataset. That makes " + str(len(embed_data)/2) + " text pairs.")
            create_tsne_figure(embed_data, dataset)
        
        embed_data = [e for e in embed_data if e[PARAPHRASE] == True]
        if len(embed_data) > 1:
            print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + " dataset. That makes " + str(len(embed_data)/2) + " text pairs.")
            create_tsne_figure(embed_data, dataset+"_paraphrasedOnly")

    else:
        # Create visualization per dataset (paraphrase & non-paraphrase pairs combined)
        print("Creating visualizations for dataset...")
        embed_data = embed_dict[dataset][EMBEDDINGS]
        if len(embed_data) > 1:
            print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + " dataset. That makes " + str(len(embed_data)/2) + " text pairs.")
            #stats_dict[dataset]["total_pairs"] = stats_dict[dataset]["total_pairs"] + int(len(embed_data)/2)
            create_tsne_figure(embed_data, dataset)

        # Create visualization per dataset (paraphrase pairs only)
        print("Creating visualizations for dataset (paraphrase-pairs only)...")
        # Filter the data to only contain paraphrased pairs:
        embed_data = [e for e in embed_data if e[PARAPHRASE] == True]
        if len(embed_data) > 1:
            print("Visualizing " + str(len(embed_data)) + " texts for the " + dataset + " dataset. That makes " + str(len(embed_data)/2) + " text pairs.")
            #stats_dict[dataset]["paraphrase_pairs"] = stats_dict[dataset]["paraphrase_pairs"] + int(len(embed_data)/2)
            create_tsne_figure(embed_data, dataset+"_paraphrasedOnly")

    #return stats_dict

def calculate_cosine_dists(df, embed_dict, dataset):
    # Calculate cosine distance of embeddings between each pair
    print("\nCalculating cosine distances between pairs embeddings...")
    embed_data = embed_dict[dataset][EMBEDDINGS]
    dataset_embeddings = {}
    for d in embed_data:
        dataset_embeddings[d[TEXT_ID]] = {EMBED: d[EMBED]}

    #dataset_embeddings = [[d[EMBED], d[TEXT_ID]] for d in embed_data]
    pair_ids = list(set([d[PAIR_ID] for d in embed_data]))

    for pair_id in tqdm(pair_ids):
        id_1 = df.loc[df[PAIR_ID] == pair_id][ID1].item()
        id_2 = df.loc[df[PAIR_ID] == pair_id][ID2].item()
        emb_1 = dataset_embeddings[id_1][EMBED]
        emb_2 = dataset_embeddings[id_2][EMBED]
        emb_1 = torch.tensor(emb_1, device = "cpu")   # move tensors to cpu
        emb_2 = torch.tensor(emb_2, device = "cpu")   
        #for d in dataset_embeddings:
        #    if d[1] == id_1:
        #        emb_1 = d[0]
        #    elif d[1] == id_2:
        #        emb_2 = d[0]
        #    if emb_1 and emb_2:
        #        break
        df.loc[df[PAIR_ID] == pair_id, COSINE_DISTANCE] = 1 - cosine(emb_1, emb_2)

    return df

def track_stats(embed_dict, dataset, stats_dict):
    embed_data = embed_dict[dataset][EMBEDDINGS]
    stats_dict[dataset]["total_pairs"] = stats_dict[dataset]["total_pairs"] + int(len(embed_data)/2)

    embed_data = [e for e in embed_data if e[PARAPHRASE] == True]
    stats_dict[dataset]["paraphrase_pairs"] = stats_dict[dataset]["paraphrase_pairs"] + int(len(embed_data)/2)
    stats_dict[dataset]["original_pairs"] = stats_dict[dataset]["total_pairs"] - int(len(embed_data)/2)

    return stats_dict

def mean_cos_distance(df, stats_dict):
    for dataset in DATASETS:
        # Calculate the mean distances per dataset (non-paraphrases and paraphrases seperately)
        df_filtered = df[(df[DATASET] == dataset) & (df[PARAPHRASE] == True)]
        mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
        print("The mean cosine distance of paraphrased pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))
        stats_dict[dataset]["mean_cos_paraphrased"] = mean_cos_dist

        df_filtered = df[(df[DATASET] == dataset) & (df[PARAPHRASE] == False)]
        mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
        print("The mean cosine distance of non-paraphrased (original) pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))
        stats_dict[dataset]["mean_cos_original"] = mean_cos_dist

        df_filtered = df[(df[DATASET] == dataset)]
        mean_cos_dist = df_filtered[COSINE_DISTANCE].mean()
        print("The mean cosine distance of all pairs (Dataset: " + dataset + ") : " + str(mean_cos_dist))
        stats_dict[dataset]["mean_cos_mixed"] = mean_cos_dist

    return stats_dict


if __name__ == "__main__":

    df = pd.read_json(os.path.join(OUT_DIR, FORMATTED_DATA_FILENAME), orient = "index")
    df[COSINE_DISTANCE] = None

    # Create stats dict
    stats_dict = {}
    for dataset in DATASETS:
        stats_dict[dataset] = { "total_pairs": 0, "paraphrase_pairs": 0, "original_pairs": 0, "mean_cos_paraphrased": 0, "mean_cos_original": 0, "mean_cos_mixed": 0 }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True ).to(device)
    # feed-forward only
    model.eval()

    print("GPU available: " +  str(torch.cuda.is_available()))
    print("Model running on " + device)

    print("Shuffle the dataframe...")
    new_df = pd.DataFrame()
    for dataset in DATASETS:
        d_df = df[df[DATASET] == dataset]   # shuffle while keeping the order of datasets in the df
        d_df = shuffle(d_df)
        new_df = pd.concat([new_df, d_df])
    df = new_df.reset_index(drop=True)
    print("Shuffled.")
    print("Total pairs found: " + str(df.shape[0]))

    #DATASETS = ["APT"]
    '''
    print("Welcome! Do you want to embed all datasets or a specific one?")
    print('1 - all datasets \n2 - a specific dataset')
    x = int(input("Enter a number: "))
    if x == 2:
        while True:
            print('Please enter the dataset you want to process (folder name). You can enter multiple datasets by seperating them by \",\" (no spaces).')
            x = input("Dataset: ").split(",")
            if all(x_item in DATASETS for x_item in x):
                print("Okay.")
                DATASETS = x
                break
            else:
                print("This is not a valid dataset. Please use the correct casing. Example: \"ETPC\" or \"SAv2\" or \"ETPC,SAv2\".")
    '''

    embed_dict = { }
    embed_dict_total = { }
    tokenized_texts = { }
    tokenized_pairs = { }
    visualized_datasets = []

    print("Tokenize texts and prepare data for embedding generation...")
    for i, row in tqdm(df[df[DATASET].isin(DATASETS)].iterrows(), total=df[df[DATASET].isin(DATASETS)].shape[0]):
        # skip invalid entries in the datasets
        if not row[TEXT1] or not row[TEXT2]:
            print("\nPair " + str(row[PAIR_ID]) + " contains empty text. Excluding pair from embeddings.")
            continue

        # mark the text with BERTs special characters
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
    last_index = 0
    skipped_counter = 0

    print("Creating embeddings for each sentence (text1 & text2) ...")
    for i, pair_id in tqdm(enumerate(list(tokenized_pairs.keys()))):

        dataset = tokenized_pairs[pair_id][DATASET]

        # handle visualization & output if neccessary (many processed or all of dataset processed)
        if dataset != last_dataset_viewed:
            if last_dataset_viewed not in visualized_datasets:
                visualize_embeddings(embed_dict, last_dataset_viewed)
                visualized_datasets.append(last_dataset_viewed)

            df = calculate_cosine_dists(df, embed_dict, last_dataset_viewed)
            stats_dict = track_stats(embed_dict, last_dataset_viewed, stats_dict)

            df[df[DATASET] == last_dataset_viewed].to_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, last_dataset_viewed+"_embedded.json"), orient = "index", index = True, indent = 4)
            last_dataset_viewed = dataset
            last_index = i

           # pop out unneccessary embeds (for final total tsne-figure)
            for d in DATASETS:
                if d in embed_dict:
                    embed_dict[d][EMBEDDINGS] = embed_dict[d][EMBEDDINGS][:2*int(FIGURE_SIZE/len(DATASETS))]    # only leave a certain amount of pairs for this dataset in total figure
            embed_dict_total = dict(embed_dict_total, **embed_dict) # update total dict

            embed_dict = { }
            skipped_counter = 0
            gc.collect()

        elif i - skipped_counter - last_index >= FIGURE_SIZE and dataset == last_dataset_viewed:     # specified figure size (max) to avoid high memory usage
            if last_dataset_viewed not in visualized_datasets:
                visualize_embeddings(embed_dict, dataset)
                visualized_datasets.append(last_dataset_viewed)

            last_index = i
            skipped_counter = 0
            gc.collect()

        # throw out longer that 512 token texts because BERT model struggels to process them
        if len(tokenized_pairs[pair_id][TOKENS1]) > 512 or len(tokenized_pairs[pair_id][TOKENS2]) > 512:
            del tokenized_pairs[pair_id]
            skipped_counter = skipped_counter + 1
            continue

        # DO FOR FIRST TEXT
        # map tokens to vocab indices
        indexed = tokenizer.convert_tokens_to_ids(tokenized_pairs[pair_id][TOKENS1])
        segments_ids = [1] * len(tokenized_pairs[pair_id][TOKENS1])
        #Extract Embeddings
        tensor = torch.tensor([indexed]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)
        # collect all the hidden states produced from all layers
        with torch.no_grad():
            hidden_states = model(tensor, segments_tensors)[2]
        # Concatenate the tensors for all layers (create a new dimension in the tensor)
        embeds = torch.stack(hidden_states, dim=0).to(device)
        # Remove dimension 1, the "batches".
        embeds = torch.squeeze(embeds, dim=1).to(device)
        #Switch dimensions
        embeds = embeds.permute(1, 0, 2)
        # Create Sentence Vector Representations (average of all token vectors)
        embedding_1 = torch.mean(hidden_states[-2][0], dim=0).to(device)

        # DO FOR SECOND TEXT
        indexed = tokenizer.convert_tokens_to_ids(tokenized_pairs[pair_id][TOKENS2])
        segments_ids = [1] * len(tokenized_pairs[pair_id][TOKENS2])
        tensor = torch.tensor([indexed]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)
        with torch.no_grad():
            hidden_states = model(tensor, segments_tensors)[2]
        embeds = torch.stack(hidden_states, dim=0).to(device)
        embeds = torch.squeeze(embeds, dim=1).to(device)
        embeds = embeds.permute(1,0,2)
        embedding_2 = torch.mean(hidden_states[-2][0], dim=0).to(device)

        # Add embeddings to dataset-specific lists:
        if dataset in embed_dict:
            embed_dict[dataset][EMBEDDINGS].append({ DATASET: dataset, PAIR_ID: pair_id, TEXT_ID: tokenized_pairs[pair_id][ID1], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW1], EMBED: list(embedding_1), PARAPHRASE: tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: False })
            embed_dict[dataset][EMBEDDINGS].append({ DATASET: dataset, PAIR_ID: pair_id, TEXT_ID: tokenized_pairs[pair_id][ID2], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW2], EMBED: list(embedding_2), PARAPHRASE: tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: True })
        else:
            embed_dict[dataset] = { EMBEDDINGS: [{ DATASET: dataset, PAIR_ID: pair_id, TEXT_ID: tokenized_pairs[pair_id][ID1], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW1], EMBED: list(embedding_1), PARAPHRASE: tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: False }] }
            embed_dict[dataset][EMBEDDINGS].append({ DATASET: dataset, PAIR_ID: pair_id, TEXT_ID: tokenized_pairs[pair_id][ID2], TEXT_PREVIEW: tokenized_pairs[pair_id][TEXT_PREVIEW2], EMBED: list(embedding_2), PARAPHRASE: tokenized_pairs[pair_id][PARAPHRASE], TUPLE_ID: True })

    # Also do after the for loop completed (for the last dataset)
    if dataset not in visualized_datasets:
        visualize_embeddings(embed_dict, dataset)
        visualized_datasets.append(dataset)

    df = calculate_cosine_dists(df, embed_dict, dataset)
    stats_dict = track_stats(embed_dict, dataset, stats_dict)

    df[df[DATASET] == dataset].to_json(os.path.join(OUT_DIR, EMBEDDINGS_FOLDER, dataset+"_embedded.json"), orient = "index", index = True, indent = 4)

    # pop out unnecessary embeds (for final total tsne-figure)
    for d in DATASETS:
        if d in embed_dict:
            embed_dict[d][EMBEDDINGS] = embed_dict[d][EMBEDDINGS][:2*int(FIGURE_SIZE/len(DATASETS))]    # only leave a certain amount of pairs for this dataset in total figure
    embed_dict_total = dict(embed_dict_total, **embed_dict) # update total dict

    # visualize all datasets in one tsne-figure
    visualize_embeddings(embed_dict_total, "total")

    # get mean cos distance per dataset
    stats_dict = mean_cos_distance(df, stats_dict)

    # Output Stats
    with open(os.path.join(OUT_DIR, 'stats_embedding_handler.json'), 'w') as f:
        json.dump(stats_dict, f, indent=4)