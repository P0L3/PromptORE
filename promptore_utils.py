import numpy as np
import torch
import pandas as pd

################################################################################
# Metrics
################################################################################
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, \
    homogeneity_score, completeness_score

def bcubed(targets, predictions, beta: float = 1):
    """B3 metric (see Baldwin1998)
    Args:
        targets (torch.Tensor): true labels
        predictions (torch.Tensor): predicted labels
        beta (float, optional): beta for f_score. Defaults to 1.
    Returns:
        Tuple[float, float, float]: b3 f1, precision and recall
    """

    cont_mat = contingency_matrix(targets, predictions)
    cont_mat_norm = cont_mat / cont_mat.sum()

    precision = np.sum(cont_mat_norm * (cont_mat /
                       cont_mat.sum(axis=0))).item()
    recall = np.sum(cont_mat_norm * (cont_mat /
                    np.expand_dims(cont_mat.sum(axis=1), 1))).item()
    f1_score = (1 + beta) * precision * recall / (beta * (precision + recall))

    return f1_score, precision, recall


def v_measure(targets, predictions):
    """V-measure
    Args:
        targets (torch.Tensor): true labels
        predictions (torch.Tensor): predictions
    Returns:
        Tuple[float, float, float]: V-measure f1, homogeneity (~prec), completeness (~rec)
    """
    homogeneity = homogeneity_score(targets, predictions)
    completeness = completeness_score(targets, predictions)
    v = 2 * homogeneity * completeness / (homogeneity + completeness)

    return v, homogeneity, completeness


def evaluate_promptore(fewrel: pd.DataFrame, predicted_labels: torch.Tensor) -> tuple:
    """Evaluate PromptORE
    Args:
        fewrel (pd.DataFrame): fewrel
        predicted_labels (torch.Tensor): predicted labels

    Returns:
        tuple: scores
    """
    labels = torch.Tensor(fewrel['output_label'].tolist()).long()

    ari = adjusted_rand_score(labels, predicted_labels)
    v, v_hom, v_comp = v_measure(labels, predicted_labels)
    b3, b3_prec, b3_rec = bcubed(labels, predicted_labels)

    return b3, b3_prec, b3_rec, v, v_hom, v_comp, ari


################################################################################
# Dataset
################################################################################
import json
from DATA.relations import relation_mapping

def parse_fewrel(path: str, expand: bool = False) -> pd.DataFrame:
    """Parse fewrel dataset. Dataset can be downloaded at:
        https://github.com/thunlp/FewRel/tree/master/data
    Args:
        path (str): path to json fewrel file
        expand (bool, Optional): to expand every instance (entity mentionned twice in
                                sentence -> 2 instances). Defaults to False.
    Returns:
        pd.DataFrame: parsed fewrel dataset
    """
    with open(path, 'r', encoding='utf-8') as file:
        fewrel_json = json.load(file)

    fewrel_tuples = []
    for relation, instances in fewrel_json.items():
        for instance in instances:
            for i_h, h_pos in enumerate(instance['h'][2]):
                for i_t, t_pos in enumerate(instance['t'][2]):
                    fewrel_tuples.append({
                        'tokens': instance['tokens'],
                        'r': relation,
                        'h': instance['h'][0],
                        'h_id': instance['h'][1],
                        'h_count': i_h,
                        'h_start': h_pos[0],
                        'h_end': h_pos[len(h_pos) - 1],
                        't': instance['t'][0],
                        't_id': instance['t'][1],
                        't_count': i_t,
                        't_start': t_pos[0],
                        't_end': t_pos[len(t_pos) - 1],
                    })
                    if not expand:
                        break
                if not expand:
                    break
    return pd.DataFrame(fewrel_tuples)

def parse_wikiphi3(path: str, expand: bool = False) -> pd.DataFrame:
    """
    0   sentence       
    1   source_name    
    2   relation_name  
    3   target_name    
    4   triple                     
    """
    # Load data
    wikiphi3 = pd.read_pickle(path)
    print("Data len: ", len(wikiphi3))
    
    # Fix column naming
    wikiphi3.rename(columns={
        "sentence": "sent"}, inplace=True)
    
    # Add e1 and e2
    def triple_entity_extract(triple):
        return triple[0], triple[1], triple[2]
    
    wikiphi3[["e1", "r", "e2"]] = wikiphi3.apply(lambda x: pd.Series(triple_entity_extract(x["triple"])), axis=1)
    wikiphi3.fillna("", inplace=True)
    
    # Remove outlayers
    
    q = wikiphi3["sent"].apply(len).quantile(q=0.999)
    wikiphi3 = wikiphi3[wikiphi3["sent"].apply(len) <= q]
    
    print("Data len final: ", len(wikiphi3))

    return wikiphi3[::20]
    

def parse_labelstudio(path: str, expand: bool = False) -> pd.DataFrame:
    
    file_path = path# "DATA/project-6-at-2025-04-22-13-14-67864b63.json"
    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
        
    ls_tuples = []    
    for entry in dataset:
        # print(entry)
        annotations = entry.get("annotations", [])
        data = entry.get("data", [])

        
        for annotation in annotations:
            entities = {e["id"]: e["value"]["text"] for e in annotation.get("result", []) if e["type"] == "labels"}
            # entity_types_mapping = {e["id"]: e["value"]["labels"] for e in annotation.get("result", []) if e["type"] == "labels"}
            relations = [r for r in annotation.get("result", []) if r["type"] == "relation"]
            
            for relation in relations:
                from_id = relation["from_id"]
                to_id = relation["to_id"]
                direction = relation["direction"] # Add ht according to direction
                
                relation_type = relation.get("labels", [""])[0]  # Extract first label or empty string
                if relation_type == "":
                    relation_type = "0"
                    
                from_node_t = entities.get(from_id, "")
                to_node_t = entities.get(to_id, "")
                
                relation_name = relation_mapping.get(int(relation_type), "")
                
                bi = False
                
                if direction == "right":
                    from_node = from_node_t
                    to_node = to_node_t
                elif direction == "left":
                    from_node = to_node_t
                    to_node = from_node_t
                else:
                    bi = True
                    from_node = from_node_t
                    to_node = to_node_t
                
                ls_tuples.append({
                        'sent': data["sentence"],
                        'r': relation_name,
                        'e1': from_node,
                        'e2': to_node,
                        'paper_id': data["paper_id"],
                        'sentence_id': data["sentence_id"]
                    })
                
                if bi:
                    ls_tuples.append({
                        'sent': data["sentence"],
                        'r': relation_name,
                        'e1': to_node,
                        'e2': from_node,
                        'paper_id': data["paper_id"],
                        'sentence_id': data["sentence_id"]
                    })
                
    
    return pd.DataFrame(ls_tuples)

################################################################################
# PromptORE
################################################################################

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from tqdm.auto import tqdm
import pickle
import ADDLIB.RelationEmbeddings.src.main.python.relcore.pre_trained_encoders.relation_encoder as relation_enc


def to_device(data, device):
    """Move data to device
    Args:
        data (Any): data
        device (str): device
    Returns:
        Any: moved data
    """
    def to_device_dict_(k: str, v, device: str):
        """Util function to move a dict (ignore variables ending with '_')
        Args:
            k (str): key
            v (Any): value
            device (str): device
        Returns:
            Any: moved value
        """
        if k.endswith('_'):
            return v
        else:
            return to_device(v, device)

    if isinstance(data, tuple):
        data = (to_device(e, device) for e in data)
    elif isinstance(data, list):
        data = [to_device(e, device) for e in data]
    elif isinstance(data, dict):
        data = {k: to_device_dict_(k, v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.to(device)

    return data


def tokenize(tokenizer, text: str, max_len: int) -> tuple:
    """Tokenize input text
    Args:
        tokenizer (any): BertTokenizer (or RobertaTokenizer)
        text (str): text to tokenize
        max_len (int): max nb of tokens

    Returns:
        tuple: input ids and attention masks
    """
    encoded_dict = tokenizer.encode_plus(
        text,                           # Sentence to encode.
        add_special_tokens=True,        # Add '[CLS]' and '[SEP]'
        max_length=max_len,        # Pad & truncate all sentences.
        padding='max_length',
        truncation=True,
        return_attention_mask=True,     # Construct attn. masks.
        return_tensors='pt',            # Return pytorch tensors.
    )
    input_ids = encoded_dict['input_ids'].view(-1)
    attention_mask = encoded_dict['attention_mask'].view(-1)
    return input_ids, attention_mask


def compute_promptore_relation_embedding(fewrel: pd.DataFrame, \
    template: str = '{e1} [MASK] {e2}.', max_len=128, device: str = 'cuda', data = "ls", emb = 1) -> pd.DataFrame:
    """Compute PromptORE relation embedding for the dataframe

    Args:
        fewrel (pd.DataFrame): fewrel dataset
        template (str, optional): template to use. 
            Authorized parameters are {e1} {e2} {sent}. Defaults to '{e1} [MASK] {e2}.'.
        max_len (int, optional): max nb of tokens. Defaults to 128.
        device (str, optional): Pytorch device to use. Defaults to cuda

    Returns:
        pd.DataFrame: fewrel dataset with relation embeddings
    """
    fewrel = fewrel.copy()
    # Setup tokenizer + bert
    ## TOKENIZER
    if emb in [1, 5]:
        tokenizer = BertTokenizer.from_pretrained(
            'P0L3/clirebert_clirevocab_uncased', do_lower_case=True)
        mask_id = tokenizer.mask_token_id

        
    ## MODEL
    if emb == 1:
        bert = BertModel.from_pretrained(
            'P0L3/clirebert_clirevocab_uncased', output_attentions=False)
    elif emb in [2, 3, 4]:
        """
        bert_model.start_of_head_entity,
        bert_model.end_of_head_entity,
        bert_model.start_of_tail_entity,
        bert_model.end_of_tail_entity,
        bert_model.mask_token,
        """
        bert = relation_enc.RelationEncoder.from_pretrained("fmmka/rel-emb-bert-b-uncased")
        tokenizer = bert.tokenizer
        
        e1 = bert.start_of_head_entity
        e2 = bert.start_of_tail_entity
        mask = bert.mask_token
        
    elif emb == 5:
        bert = BertForMaskedLM.from_pretrained(
            'P0L3/clirebert_clirevocab_uncased', output_attentions=False)

    # Tokenize fewrel
    rows = []
    for _, instance in tqdm(fewrel.iterrows(), total=len(fewrel)):
        if data == "fewrel":
            tokens = instance['tokens'].copy()
            head = ' '.join(tokens[instance['h_start']:instance['h_end']+1])
            tail = ' '.join(tokens[instance['t_start']:instance['t_end']+1])

            sent = ' '.join(tokens)
        elif data in ["wikiphi3", "ls"]:
            head = instance["e1"]
            tail = instance["e2"]
            sent = instance["sent"]
            
        text = template.format(e1=head, e2=tail, sent=sent)

        input_ids, attention_mask = tokenize(tokenizer, text, max_len)
        
        try:
            rows.append({
                'input_tokens': input_ids,
                'input_attention_mask': attention_mask,
                'input_mask': (input_ids == mask_id).nonzero().flatten().item(),
                'output_r': instance['r'],
                'head': head,
                'tail': tail,
                "sentence": sent
            })
        except ValueError:
            print(instance)
        # print(len(input_ids))

    complete_fewrel = pd.DataFrame(rows)
    complete_fewrel['output_label'] = pd.factorize(
        complete_fewrel['output_r'])[0]

    # Predict embeddings
    bert.to(device)
    bert.eval()

    tokens = torch.stack(complete_fewrel['input_tokens'].tolist(), dim=0)
    attention_mask = torch.stack(
        complete_fewrel['input_attention_mask'].tolist(), dim=0)
    masks = torch.Tensor(complete_fewrel['input_mask'].tolist()).long()
    dataset = TensorDataset(tokens, attention_mask, masks)
    dataloader = DataLoader(dataset, num_workers=1,
                            batch_size=24, shuffle=False)
    if emb == 1: # Original embeddings from PromptORE paper
        with torch.no_grad():
            embeddings = []
            for batch in tqdm(dataloader):
                tokens, attention_mask, mask = batch
                tokens = tokens.to(device)
                attention_mask = attention_mask.to(device)
                out = bert(tokens, attention_mask)[0].detach()     
                arange = torch.arange(out.shape[0])
                embedding = out[arange, mask]
                
                embeddings.append(embedding)
                del out
            embeddings = torch.cat(embeddings, dim=0).detach().to('cpu')
            embeddings_list = list(embeddings)
            with open("embeddings_mask_only_sms.pkl", "wb") as f:
                pickle.dump(embeddings_list[:10], f)
    elif emb == 2:  # E1 embeddings
        with torch.no_grad():
            embeddings = []
            for batch in tqdm(dataloader):
                tokens, attention_mask, e1 = batch
                tokens = tokens.to(device)
                attention_mask = attention_mask.to(device)
                out = bert(tokens, attention_mask)[0].detach()
                arange = torch.arange(out.shape[0])
                embedding = out[arange, e1]
                embeddings.append(embedding)
                del out
            embeddings = torch.cat(embeddings, dim=0).detach().to('cpu')
            embeddings_list = list(embeddings)
            with open("embeddings_e1_only_sms.pkl", "wb") as f:
                pickle.dump(embeddings_list[:10], f)

    elif emb == 3:  # E2 embeddings
        with torch.no_grad():
            embeddings = []
            for batch in tqdm(dataloader):
                tokens, attention_mask, _, e2 = batch
                tokens = tokens.to(device)
                attention_mask = attention_mask.to(device)
                out = bert(tokens, attention_mask)[0].detach()
                arange = torch.arange(out.shape[0])
                embedding = out[arange, e2]
                embeddings.append(embedding)
                del out
            embeddings = torch.cat(embeddings, dim=0).detach().to('cpu')
            embeddings_list = list(embeddings)
            with open("embeddings_e2_only_sms.pkl", "wb") as f:
                pickle.dump(embeddings_list[:10], f)

    elif emb == 4:  # Concatenate E1, E2, and MASK embeddings
        with torch.no_grad():
            embeddings = []
            for batch in tqdm(dataloader):
                tokens, attention_mask, mask, e1, e2 = batch
                tokens = tokens.to(device)
                attention_mask = attention_mask.to(device)
                out = bert(tokens, attention_mask)[0].detach()
                arange = torch.arange(out.shape[0])
                e1_emb = out[arange, e1]
                e2_emb = out[arange, e2]
                mask_emb = out[arange, mask]
                combined = torch.cat([e1_emb, e2_emb, mask_emb], dim=1)
                embeddings.append(combined)
                del out
            embeddings = torch.cat(embeddings, dim=0).detach().to('cpu')
            embeddings_list = list(embeddings)
            with open("embeddings_e1_e2_mask_concat_sms.pkl", "wb") as f:
                pickle.dump(embeddings_list[:10], f)            
    
    elif emb == 5: # First N embeddings concatenated
        TOP_N = 1
        with torch.no_grad():
            embeddings = []

            for batch in tqdm(dataloader):
                tokens, attention_mask, mask = batch
                tokens = tokens.to(device)
                attention_mask = attention_mask.to(device)
                mask = mask.to(device)

                # Step 1: Get logits and top-N predictions at [MASK] position
                output = bert(input_ids=tokens, attention_mask=attention_mask)
                logits = output.logits  # [batch_size, seq_len, vocab_size]
                vocab_probs = torch.softmax(logits, dim=-1)

                # Collect top-N token IDs for each input
                batch_size = tokens.size(0)
                top_embeddings = []

                for i in range(batch_size):
                    mask_pos = mask[i].item()  # position of [MASK] in this sequence
                    top_n_tokens = torch.topk(vocab_probs[i, mask_pos], k=TOP_N).indices  # [TOP_N]

                    concat_embedding = []
                    for token_id in top_n_tokens:
                        modified_input = tokens[i].clone()
                        modified_input[mask_pos] = token_id  # Replace [MASK] with predicted token

                        # Forward pass again on modified input
                        out_mod = bert(input_ids=modified_input.unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), output_hidden_states=True).hidden_states[-1]
                        concat_embedding.append(out_mod[0, mask_pos])  # Embedding at replaced token position

                    # Concatenate top-N embeddings
                    final_embed = torch.cat(concat_embedding, dim=-1)  # shape: [TOP_N * hidden_dim]
                    top_embeddings.append(final_embed)

                batch_embed = torch.stack(top_embeddings, dim=0)
                embeddings.append(batch_embed)

            embeddings = torch.cat(embeddings, dim=0).detach().to('cpu')
            embeddings_list = list(embeddings)
            with open("embeddings_topk1_sms.pkl", "wb") as f:
                pickle.dump(embeddings_list[:10], f)

    bert.to('cpu')
    complete_fewrel['embedding'] = embeddings_list
    return complete_fewrel


def compute_kmeans_clustering(fewrel_relation_embeddings: pd.DataFrame, n_rel: int, \
    random_state: int):
    """Compute kmeans clustering with fixed nb of clusters
    Args:
        fewrel_relation_embeddings (pd.DataFrame): relation embeddings
        n_rel (int): number of relations (nb of clusters)
    Returns:
        torch.Tensor: predicted labels
    """
    embeddings = torch.stack(fewrel_relation_embeddings['embedding'].tolist())

    model = KMeans(init='k-means++', n_init=10, n_clusters=n_rel, random_state=random_state)
    predicted_labels = model.fit(embeddings)
    predicted_labels = model.predict(embeddings)

    return predicted_labels


def estimate_n_rel(fewrel_relation_embeddings: pd.DataFrame, random_state: int, \
    k_range: tuple = [10, 300], k_step: int = 5) -> int:
    """Estimate number of clusters using the elbow rule

    Args:
        fewrel_relation_embeddings (pd.DataFrame): relation embeddings
        k_range (tuple, optional): range of clusters to test. Defaults to [10, 300].
        k_step (int, optional): step. Defaults to 5.

    Returns:
        int: estimated number of clusters
    """
    embeddings = torch.stack(fewrel_relation_embeddings['embedding'].tolist())

    ks = np.arange(k_range[0], k_range[1], k_step)
    model = KMeans(init='k-means++', n_init=10, random_state=random_state)
    visualizer = KElbowVisualizer(
        model, k=ks, metric='silhouette', timings=False, locate_elbow=False)
    visualizer.fit(embeddings)
    silhouette = pd.DataFrame()
    silhouette['ks'] = ks
    silhouette['scores'] = visualizer.k_scores_

    # Kernel ridge
    model = KernelRidge(kernel='rbf', degree=3, gamma=1e-3)
    X = silhouette['ks'].values.reshape(-1, 1)
    model.fit(X=X, y=silhouette['scores'])
    p = model.predict(X=X)

    k_elbow = silhouette['ks'][p.argmax()]
    return k_elbow
