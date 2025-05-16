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
import re

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
    
def process_row(row, 
                e1_start_marker: str, 
                e1_end_marker: str, 
                e2_start_marker: str, 
                e2_end_marker: str, 
                mask_token: str,
                r = False):
    
    sent, e1, e2 = row["sent"], str(row["e1"]), str(row["e2"])
    
        

    # Tag entities with provided markers
    sent_t = re.sub(re.escape(e1), f"{e1_start_marker}{e1}{e1_end_marker}", sent)
    sent_t1 = re.sub(re.escape(e2), f"{e2_start_marker}{e2}{e2_end_marker}", sent_t)

    # Append relation question with mask token
    sent_t2 = f"{sent_t1} The relation between {e1} and {e2} is {mask_token}."
    
    if r:
        sent_t2 = e1 + " " + row["r"] + " " + e2 + "." + sent_t2
        
    return sent_t2

def parse_wikiphi3_with_dynamic_markers(
    path: str, 
    e1_start_marker: str, 
    e1_end_marker: str, 
    e2_start_marker: str, 
    e2_end_marker: str, 
    mask_token: str,
    expand: bool = False) -> pd.DataFrame:
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
    
    wikiphi3["sent"] = wikiphi3.apply(
        lambda row: process_row(
            row,
            e1_start_marker, e1_end_marker,
            e2_start_marker, e2_end_marker,
            mask_token, True
        ),
        axis=1
    )

    return wikiphi3[::10]




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

# In your DATA.relations or wherever parse_labelstudio is
# from DATA.relations import relation_mapping # Already there
# import pandas as pd
# import json
# import re # Not used in your snippet, can be removed

def parse_labelstudio_with_dynamic_markers(
    path: str, 
    e1_start_marker: str, 
    e1_end_marker: str, 
    e2_start_marker: str, 
    e2_end_marker: str, 
    mask_token: str,
    expand: bool = False 
) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
        
    ls_tuples = []    
    for entry in dataset:
        annotations = entry.get("annotations", [])
        data = entry.get("data", {})
        # ... (sentence, paper_id, sentence_id extraction) ...
        sentence = data.get("sentence", "")
        paper_id = data.get("paper_id")
        sentence_id = data.get("sentence_id")

        for annotation in annotations:
            # ... (entities, relations extraction) ...
            results = annotation.get("result", [])
            entities = {e["id"]: e["value"] for e in results if e["type"] == "labels"}
            relations = [r for r in results if r["type"] == "relation"]

            for relation in relations:
                # ... (relation details extraction) ...
                from_id = relation["from_id"]
                to_id = relation["to_id"]
                direction = relation["direction"]
                relation_type = relation.get("labels", [""])[0] or "0"
                relation_name = relation_mapping.get(int(relation_type), "") # Ensure relation_mapping is defined

                e1_val_key = from_id if direction == "right" else to_id
                e2_val_key = to_id if direction == "right" else from_id
                
                # Handle case where entity might not be found if direction is "undirected" and IDs are swapped
                # or if entities dict is incomplete.
                e1_value_data = entities.get(e1_val_key, {})
                e2_value_data = entities.get(e2_val_key, {})

                e1_text = e1_value_data.get("text", "")
                e1_start = e1_value_data.get("start", -1)
                e1_end = e1_value_data.get("end", -1)

                e2_text = e2_value_data.get("text", "")
                e2_start = e2_value_data.get("start", -1)
                e2_end = e2_value_data.get("end", -1)

                # Ensure valid spans before proceeding
                if not (e1_text and e2_text and e1_start != -1 and e2_start != -1):
                    # print(f"Skipping relation due to missing entity text or spans: e1='{e1_text}', e2='{e2_text}'")
                    continue

                # Insert entity markers into the sentence
                spans = sorted([
                    (e1_start, e1_end, e1_start_marker, e1_end_marker), # Use dynamic markers
                    (e2_start, e2_end, e2_start_marker, e2_end_marker)  # Use dynamic markers
                ], key=lambda x: x[0], reverse=True)

                marked_sentence = sentence
                for start, end, start_tag, end_tag in spans:
                    if start != -1 and end != -1 : # Ensure valid span
                        marked_sentence = marked_sentence[:end] + end_tag + marked_sentence[end:]
                        marked_sentence = marked_sentence[:start] + start_tag + marked_sentence[start:]
                
                prompt = f" The relation between {e1_text} and {e2_text} is {mask_token}." # Use dynamic mask token
                full_sentence = marked_sentence + prompt

                ls_tuples.append({
                    'sent': full_sentence,
                    'r': relation_name,
                    'e1': e1_text, # original entity text
                    'e2': e2_text, # original entity text
                    'paper_id': paper_id,
                    'sentence_id': sentence_id
                })

                if direction == "undirected": # Or your specific bidirectional logic
                    # For bidirectional, ensure spans are correctly identified if you re-use marked_sentence
                    # It might be safer to re-mark the original sentence if entity roles swap
                    # For now, assuming entity markers are correctly placed and only prompt changes for reversed e1/e2
                    prompt_rev = f" The relation between {e2_text} and {e1_text} is {mask_token}."
                    # If marked_sentence structure is fixed for [E1]=original_e1, [E2]=original_e2,
                    # then for reversed relation, you might need to re-evaluate marker placement
                    # or adjust how e1_text/e2_text are used in prompt.
                    # Your current `marked_sentence` has fixed E1 and E2 markers.
                    # A simple reversal of prompt is fine if the task is symmetric or context allows.
                    full_sentence_rev = marked_sentence + prompt_rev # Uses same marked_sentence
                    ls_tuples.append({
                        'sent': full_sentence_rev,
                        'r': relation_name, # Assuming same relation name for reversed
                        'e1': e2_text, # Swapped entity text
                        'e2': e1_text, # Swapped entity text
                        'paper_id': paper_id,
                        'sentence_id': sentence_id
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
    template: str = '{E1}{e1} {MASK} {E2}{e2}.', max_len=128, device: str = 'cuda', data="ls", emb=1) -> pd.DataFrame:
    fewrel = fewrel.copy()
    
    # Variables for marker strings and their token IDs
    # These will be populated based on `emb` and `data` source
    e1_marker_str_for_template_formatting = ""  # For constructing text if not pre-formatted
    e2_marker_str_for_template_formatting = ""
    
    e1_pos_token_id_to_search = None # ID of E1 marker to find in tokenized input
    e2_pos_token_id_to_search = None # ID of E2 marker to find in tokenized input
    mask_token_id_to_search = None   # ID of MASK to find in tokenized input

    bert_for_embeddings = None 

    if emb == 1:
        tokenizer = BertTokenizer.from_pretrained('P0L3/clirebert_clirevocab_uncased', do_lower_case=True)
        bert_for_embeddings = BertModel.from_pretrained('P0L3/clirebert_clirevocab_uncased', output_attentions=False)
        mask_token_id_to_search = tokenizer.mask_token_id
        # e1/e2 markers are not explicitly searched for emb=1 in this setup
        
    elif emb in [2, 3, 4]:
        if relation_enc is None:
            raise ModuleNotFoundError("relation_enc module (from ADDLIB) not loaded. Cannot use emb=2,3,4.")
        
        bert_rel_model = relation_enc.RelationEncoder.from_pretrained("fmmka/rel-emb-bert-b-uncased")
        tokenizer = bert_rel_model.tokenizer 
        bert_for_embeddings = bert_rel_model 
        
        # These are the FMMKA model's actual special token strings
        model_specific_e1_marker = bert_rel_model.start_of_head_entity
        model_specific_e2_marker = bert_rel_model.start_of_tail_entity
        model_specific_mask_token = bert_rel_model.mask_token # This is also tokenizer.mask_token

        # For template formatting (if data is not "ls")
        e1_marker_str_for_template_formatting = model_specific_e1_marker
        e2_marker_str_for_template_formatting = model_specific_e2_marker
        
        if data in ["ls", "wikiphi3"]:
            # Your parse_labelstudio uses hardcoded "[E1]", "[E2]", "[MASK]" (ignoring spaces for token ID lookup for now)
            # We must find the token IDs of THESE strings using the FMMKA tokenizer.
            ls_hardcoded_e1_marker = "[E1]" # From your parse_labelstudio
            ls_hardcoded_e2_marker = "[E2]" # From your parse_labelstudio
            ls_hardcoded_mask_marker = "[MASK]" # From your parse_labelstudio's prompt

            e1_pos_token_id_to_search = tokenizer.convert_tokens_to_ids(ls_hardcoded_e1_marker)
            e2_pos_token_id_to_search = tokenizer.convert_tokens_to_ids(ls_hardcoded_e2_marker)
            mask_token_id_to_search = tokenizer.convert_tokens_to_ids(ls_hardcoded_mask_marker)

            if e1_pos_token_id_to_search == tokenizer.unk_token_id: 
                print(f"Warning: LabelStudio's hardcoded E1 marker '{ls_hardcoded_e1_marker}' is UNK for the fmmka tokenizer.")
            if e2_pos_token_id_to_search == tokenizer.unk_token_id: 
                print(f"Warning: LabelStudio's hardcoded E2 marker '{ls_hardcoded_e2_marker}' is UNK for the fmmka tokenizer.")
            if mask_token_id_to_search == tokenizer.unk_token_id:
                 print(f"Warning: LabelStudio's hardcoded MASK marker '{ls_hardcoded_mask_marker}' is UNK for the fmmka tokenizer. This might cause issues if different from model's actual mask token '{model_specific_mask_token}'.")
        else: # For "fewrel", "wikiphi3" etc.
            e1_pos_token_id_to_search = tokenizer.convert_tokens_to_ids(model_specific_e1_marker)
            e2_pos_token_id_to_search = tokenizer.convert_tokens_to_ids(model_specific_e2_marker)
            mask_token_id_to_search = tokenizer.mask_token_id # Use the model's official mask token ID

            if e1_pos_token_id_to_search == tokenizer.unk_token_id: print(f"Warning: Model's E1 marker '{model_specific_e1_marker}' is UNK for its own tokenizer.")
            if e2_pos_token_id_to_search == tokenizer.unk_token_id: print(f"Warning: Model's E2 marker '{model_specific_e2_marker}' is UNK for its own tokenizer.")
            
    elif emb == 5:
        tokenizer = BertTokenizer.from_pretrained('P0L3/clirebert_clirevocab_uncased', do_lower_case=True)
        bert_for_embeddings = BertForMaskedLM.from_pretrained(
            'P0L3/clirebert_clirevocab_uncased', output_attentions=False, output_hidden_states=True
        )
        mask_token_id_to_search = tokenizer.mask_token_id
    else:
        raise ValueError(f"Unsupported emb value: {emb}")

    if bert_for_embeddings is None:
        raise ValueError("BERT model for embeddings was not initialized.")
    if mask_token_id_to_search is None and emb not in [2,3]: # Mask ID is crucial unless only E1/E2 specific focus
         print(f"Warning: mask_token_id_to_search is not set for emb={emb}. This might be an issue.")


    rows = []
    for _, instance in tqdm(fewrel.iterrows(), total=len(fewrel), desc="Processing instances"):
        text_to_tokenize = ""
        if data == "fewrel":
            tokens_orig = instance['tokens'].copy()
            head = ' '.join(tokens_orig[instance['h_start']:instance['h_end']+1])
            tail = ' '.join(tokens_orig[instance['t_start']:instance['t_end']+1])
            sent_orig = ' '.join(tokens_orig) # Original sentence for context if template uses {sent}
            text_to_tokenize = template.format(
                e1=head, e2=tail, sent=sent_orig, 
                MASK=tokenizer.mask_token, # Actual string for current tokenizer's mask
                E1=e1_marker_str_for_template_formatting, # Model-specific or ""
                E2=e2_marker_str_for_template_formatting  # Model-specific or ""
            )
        elif data == "wikiphi3":
            head = instance["e1"]
            tail = instance["e2"]
            sent_orig = instance["sent"]
            text_to_tokenize = template.format(
                e1=head, e2=tail, sent=sent_orig,
                MASK=tokenizer.mask_token,
                E1=e1_marker_str_for_template_formatting,
                E2=e2_marker_str_for_template_formatting
            )
        elif data == "ls":
            head = instance["e1"] # Extracted by your parse_labelstudio
            tail = instance["e2"] # Extracted by your parse_labelstudio
            text_to_tokenize = instance["sent"] # This is pre-formatted by your new parse_labelstudio
                                             # It should contain [E1], [/E1], [E2], [/E2], and [MASK] (or model specific ones if parser is updated)
        else:
            raise ValueError(f"Unknown data type: {data}")
        
        
        input_ids, attention_mask = tokenize(tokenizer, text_to_tokenize, max_len)
        
        current_row_data = {
            'input_tokens': input_ids, 'input_attention_mask': attention_mask,
            'output_r': instance['r'], 'head': head, 'tail': tail,
            "sentence": text_to_tokenize, # Store the actual tokenized sentence
            'input_mask': -1, 'input_e1_pos': -1, 'input_e2_pos': -1
        }
        
        if mask_token_id_to_search is not None:
            mask_positions = (input_ids == mask_token_id_to_search).nonzero().flatten()
            if len(mask_positions) > 0:
                current_row_data['input_mask'] = mask_positions[0].item()

        # For emb 2,3,4, we need E1/E2 positions (using the IDs determined earlier based on data source)
        if emb in [2, 3, 4]:
            if e1_pos_token_id_to_search is not None:
                e1_marker_positions = (input_ids == e1_pos_token_id_to_search).nonzero().flatten()
                if len(e1_marker_positions) > 0:
                    current_row_data['input_e1_pos'] = e1_marker_positions[0].item()
            
            if e2_pos_token_id_to_search is not None: 
                e2_marker_positions = (input_ids == e2_pos_token_id_to_search).nonzero().flatten()
                if len(e2_marker_positions) > 0:
                    current_row_data['input_e2_pos'] = e2_marker_positions[0].item()
        
        rows.append(current_row_data)
        
    if not rows:
        print("Warning: No data rows were processed.")
        return pd.DataFrame()

    complete_fewrel = pd.DataFrame(rows)
    if complete_fewrel.empty:
         print("Warning: DataFrame is empty after processing rows.")
         return complete_fewrel

    complete_fewrel['output_label'] = pd.factorize(complete_fewrel['output_r'])[0]

    # Prepare tensors for Dataset
    # ... (Dataset and DataLoader setup remains the same as previous correct version)
    tokens_tensor = torch.stack(complete_fewrel['input_tokens'].tolist(), dim=0)
    attention_mask_tensor = torch.stack(complete_fewrel['input_attention_mask'].tolist(), dim=0)

    dataset_tensors = [tokens_tensor, attention_mask_tensor]
    if emb == 1 or emb == 5:
        dataset_tensors.append(torch.Tensor(complete_fewrel['input_mask'].tolist()).long())
    elif emb == 2:
        dataset_tensors.append(torch.Tensor(complete_fewrel['input_e1_pos'].tolist()).long())
    elif emb == 3:
        dataset_tensors.append(torch.Tensor(complete_fewrel['input_mask'].tolist()).long()) 
        dataset_tensors.append(torch.Tensor(complete_fewrel['input_e2_pos'].tolist()).long())
    elif emb == 4:
        dataset_tensors.append(torch.Tensor(complete_fewrel['input_mask'].tolist()).long())
        dataset_tensors.append(torch.Tensor(complete_fewrel['input_e1_pos'].tolist()).long())
        dataset_tensors.append(torch.Tensor(complete_fewrel['input_e2_pos'].tolist()).long())
    
    dataset = TensorDataset(*dataset_tensors)
    dataloader = DataLoader(dataset, num_workers=1, batch_size=24, shuffle=False)


    bert_for_embeddings.to(device)
    bert_for_embeddings.eval()
    
    embeddings_list = [] 

    # --- Embedding extraction loops ---
    if emb == 1: # Uses P0L3/clirebert (BertModel)
        with torch.no_grad():
            batch_embeddings_list = []
            for batch in tqdm(dataloader, desc="Embedding (emb=1)"):
                tokens_b, attention_mask_b, mask_b = batch 
                tokens_b, attention_mask_b, mask_b = tokens_b.to(device), attention_mask_b.to(device), mask_b.to(device)
                valid_indices = mask_b != -1
                if not torch.any(valid_indices): continue
                
                model_output = bert_for_embeddings(input_ids=tokens_b[valid_indices], attention_mask=attention_mask_b[valid_indices])
                out = model_output[0] 
                arange = torch.arange(out.shape[0], device=device)
                embedding = out[arange, mask_b[valid_indices]].detach()
                batch_embeddings_list.append(embedding)
            if batch_embeddings_list: embeddings_list = list(torch.cat(batch_embeddings_list, dim=0).cpu())
            with open("embeddings_mask_only_sms.pkl", "wb") as f: pickle.dump(embeddings_list[:10], f)

    elif emb in [2, 3, 4]: # Uses fmmka/rel-emb (RelationEncoder)
        with torch.no_grad():
            batch_embeddings_list = []
            desc_str = f"Embedding (emb={emb})"
            for batch in tqdm(dataloader, desc=desc_str):
                # Unpack based on what was added to dataset_tensors for this emb
                if emb == 2: tokens_b, attention_mask_b, e1_b = batch
                elif emb == 3: tokens_b, attention_mask_b, _, e2_b = batch # _ is mask_pos
                elif emb == 4: tokens_b, attention_mask_b, mask_b, e1_b, e2_b = batch

                tokens_b, attention_mask_b = tokens_b.to(device), attention_mask_b.to(device)
                if emb == 2: e1_b = e1_b.to(device)
                elif emb == 3: e2_b = e2_b.to(device)
                elif emb == 4: mask_b, e1_b, e2_b = mask_b.to(device), e1_b.to(device), e2_b.to(device)

                # Determine valid_indices based on required positions for the current emb strategy
                current_valid_indices = torch.ones(tokens_b.shape[0], dtype=torch.bool, device=device)
                if emb == 2: current_valid_indices &= (e1_b != -1)
                elif emb == 3: current_valid_indices &= (e2_b != -1)
                elif emb == 4: current_valid_indices &= (mask_b != -1) & (e1_b != -1) & (e2_b != -1)
                
                if not torch.any(current_valid_indices): continue

                # Filter all tensors based on combined valid_indices for this batch
                tokens_vf = tokens_b[current_valid_indices]
                attention_mask_vf = attention_mask_b[current_valid_indices]
                if emb == 2: e1_vf = e1_b[current_valid_indices]
                elif emb == 3: e2_vf = e2_b[current_valid_indices]
                elif emb == 4: 
                    mask_vf = mask_b[current_valid_indices]
                    e1_vf = e1_b[current_valid_indices]
                    e2_vf = e2_b[current_valid_indices]
                
                input_dict = {"input_ids": tokens_vf, "attention_mask": attention_mask_vf}
                out = bert_for_embeddings(input_dict) # Output is directly hidden states tensor
                
                arange = torch.arange(out.shape[0], device=device)
                if emb == 2: embedding = out[arange, e1_vf].detach()
                elif emb == 3: embedding = out[arange, e2_vf].detach()
                elif emb == 4:
                    e1_emb = out[arange, e1_vf]
                    e2_emb = out[arange, e2_vf]
                    mask_emb = out[arange, mask_vf]
                    embedding = torch.cat([e1_emb, e2_emb, mask_emb], dim=1).detach()
                
                batch_embeddings_list.append(embedding)

            if batch_embeddings_list: embeddings_list = list(torch.cat(batch_embeddings_list, dim=0).cpu())
            
            if emb == 2: pkl_name = "embeddings_e1_only_sms.pkl"
            elif emb == 3: pkl_name = "embeddings_e2_only_sms.pkl"
            elif emb == 4: pkl_name = "embeddings_e1_e2_mask_concat_sms.pkl"
            with open(pkl_name, "wb") as f: pickle.dump(embeddings_list[:10], f)
    
    elif emb == 5: # Uses P0L3/clirebert (BertForMaskedLM)
        TOP_N = 1
        with torch.no_grad():
            batch_embeddings_list = []
            for batch in tqdm(dataloader, desc="Embedding (emb=5)"):
                tokens_b, attention_mask_b, mask_b = batch
                tokens_b, attention_mask_b, mask_b = tokens_b.to(device), attention_mask_b.to(device), mask_b.to(device)
                valid_indices = mask_b != -1
                if not torch.any(valid_indices): continue
                
                tokens_vf, attention_mask_vf, mask_vf = tokens_b[valid_indices], attention_mask_b[valid_indices], mask_b[valid_indices]

                output_lm = bert_for_embeddings(input_ids=tokens_vf, attention_mask=attention_mask_vf) 
                logits = output_lm.logits 
                vocab_probs = torch.softmax(logits, dim=-1) 
                current_valid_batch_size = tokens_vf.size(0)
                top_embeddings_for_batch = []

                for i in range(current_valid_batch_size):
                    mask_pos_in_instance = mask_vf[i].item()
                    top_n_token_ids = torch.topk(vocab_probs[i, mask_pos_in_instance], k=TOP_N).indices
                    concat_embedding_for_instance = []
                    for token_id_replace in top_n_token_ids:
                        modified_input_ids_instance = tokens_vf[i].clone()
                        modified_input_ids_instance[mask_pos_in_instance] = token_id_replace
                        out_mod_all_layers = bert_for_embeddings(
                            input_ids=modified_input_ids_instance.unsqueeze(0), 
                            attention_mask=attention_mask_vf[i].unsqueeze(0)
                        )
                        last_hidden_state = out_mod_all_layers.hidden_states[-1]
                        concat_embedding_for_instance.append(last_hidden_state[0, mask_pos_in_instance])
                    final_embed_instance = torch.cat(concat_embedding_for_instance, dim=-1).detach()
                    top_embeddings_for_batch.append(final_embed_instance)
                
                if top_embeddings_for_batch:
                     batch_embeddings_list.append(torch.stack(top_embeddings_for_batch, dim=0))
            
            if batch_embeddings_list: embeddings_list = list(torch.cat(batch_embeddings_list, dim=0).cpu())
            with open("embeddings_topk1_sms.pkl", "wb") as f: pickle.dump(embeddings_list[:10], f)

    bert_for_embeddings.to('cpu')
    
    # Assign embeddings back to DataFrame
    if embeddings_list:
        # This logic attempts to align embeddings with rows that were expected to produce them
        valid_rows_mask = pd.Series([True] * len(complete_fewrel), index=complete_fewrel.index)
        if emb == 1 or emb == 5: valid_rows_mask &= (complete_fewrel['input_mask'] != -1)
        elif emb == 2: valid_rows_mask &= (complete_fewrel['input_e1_pos'] != -1)
        elif emb == 3: valid_rows_mask &= (complete_fewrel['input_e2_pos'] != -1) # Assumes mask can be -1
        elif emb == 4: valid_rows_mask &= ((complete_fewrel['input_mask'] != -1) & \
                                           (complete_fewrel['input_e1_pos'] != -1) & \
                                           (complete_fewrel['input_e2_pos'] != -1))
        
        target_df_for_embeddings = complete_fewrel[valid_rows_mask].copy()

        if len(embeddings_list) == len(target_df_for_embeddings):
            target_df_for_embeddings['embedding'] = embeddings_list
            print(f"Successfully generated and assigned {len(embeddings_list)} embeddings.")
            return target_df_for_embeddings
        elif embeddings_list: 
             print(f"Warning: Mismatch assigning embeddings. Generated: {len(embeddings_list)}, Expected valid rows in DataFrame: {len(target_df_for_embeddings)}. This may indicate issues in data processing or filtering.")
             # Fallback: if lengths don't match, it's risky to assign. Best to return unmerged or investigate.
             # If you are certain about a partial assignment:
             # if len(embeddings_list) <= len(target_df_for_embeddings) and len(embeddings_list) > 0 :
             #     print(f"Attempting to assign {len(embeddings_list)} embeddings to the first {len(embeddings_list)} identified valid rows.")
             #     target_df_for_embeddings = target_df_for_embeddings.iloc[:len(embeddings_list)].copy()
             #     target_df_for_embeddings['embedding'] = embeddings_list
             #     return target_df_for_embeddings
             print("Returning DataFrame; embeddings were generated but could not be reliably merged due to count mismatch.")
             return target_df_for_embeddings # Or complete_fewrel
        else: 
            print("Warning: `embeddings_list` is empty, though some rows were identified as valid for processing. No embeddings assigned.")
            return target_df_for_embeddings # Return the dataframe that was prepared for embeddings
            
    else: 
        print("Warning: No embeddings were generated at all. Returning original processed DataFrame without 'embedding' column.")
        return complete_fewrel



# def compute_promptore_relation_embedding(fewrel: pd.DataFrame, \
#     template: str = '{E1}{e1} {MASK} {E2}{e2}.', max_len=128, device: str = 'cuda', data="wikiphi3", emb=1) -> pd.DataFrame:
#     fewrel = fewrel.copy()
#     e1_marker_str, e2_marker_str = "", ""
#     e1_pos_token_id, e2_pos_token_id = None, None 
#     bert_for_embeddings = None 

#     if emb == 1:
#         tokenizer = BertTokenizer.from_pretrained('P0L3/clirebert_clirevocab_uncased', do_lower_case=True)
#         bert_for_embeddings = BertModel.from_pretrained('P0L3/clirebert_clirevocab_uncased', output_attentions=False)
#     elif emb in [2, 3, 4]:
#         if relation_enc is None:
#             raise ModuleNotFoundError("relation_enc module (from ADDLIB) not loaded. Cannot use emb=2,3,4.")
        
#         # bert_rel_model is the direct model instance from from_pretrained
#         bert_rel_model = relation_enc.RelationEncoder.from_pretrained("fmmka/rel-emb-bert-b-uncased") ## MODIFIED ##
        
#         tokenizer = bert_rel_model.tokenizer 
#         e1_marker_str = bert_rel_model.start_of_head_entity
#         e2_marker_str = bert_rel_model.start_of_tail_entity
#         print(e1_marker_str)
#         bert_for_embeddings = bert_rel_model ## MODIFIED: Use the model directly ##
        
#         e1_pos_token_id = tokenizer.convert_tokens_to_ids(e1_marker_str)
#         e2_pos_token_id = tokenizer.convert_tokens_to_ids(e2_marker_str)
        
#         if e1_pos_token_id == tokenizer.unk_token_id:
#             print(f"Warning: E1 marker string '{e1_marker_str}' (for emb={emb}) is UNK for the tokenizer.")
#         if e2_pos_token_id == tokenizer.unk_token_id:
#             print(f"Warning: E2 marker string '{e2_marker_str}' (for emb={emb}) is UNK for the tokenizer.")
#     elif emb == 5:
#         tokenizer = BertTokenizer.from_pretrained('P0L3/clirebert_clirevocab_uncased', do_lower_case=True)
#         bert_for_embeddings = BertForMaskedLM.from_pretrained(
#             'P0L3/clirebert_clirevocab_uncased', 
#             output_attentions=False, 
#             output_hidden_states=True 
#         )
#     else:
#         raise ValueError(f"Unsupported emb value: {emb}")

#     if bert_for_embeddings is None:
#         raise ValueError("BERT model for embeddings was not initialized.")

#     mask_token_id_to_find = tokenizer.mask_token_id

#     # ... (rest of the row processing and DataLoader setup remains the same) ...
#     # Tokenize data and prepare for DataLoader
#     rows = []
#     for _, instance in tqdm(fewrel.iterrows(), total=len(fewrel), desc="Processing instances"):
#         if data == "fewrel":
#             tokens_orig = instance['tokens'].copy()
#             head = ' '.join(tokens_orig[instance['h_start']:instance['h_end']+1])
#             tail = ' '.join(tokens_orig[instance['t_start']:instance['t_end']+1])
#             sent = ' '.join(tokens_orig)
#         elif data in ["wikiphi3", "ls"]:
#             head = instance["e1"]
#             tail = instance["e2"]
#             sent = instance["sent"]
#         else:
#             raise ValueError(f"Unknown data type: {data}")
            
#         text = template.format(
#             e1=head, 
#             e2=tail, 
#             sent=sent, 
#             MASK=tokenizer.mask_token,
#             E1=e1_marker_str,        
#             E2=e2_marker_str         
#         )

#         input_ids, attention_mask = tokenize(tokenizer, text, max_len)
        
#         current_row_data = {
#             'input_tokens': input_ids,
#             'input_attention_mask': attention_mask,
#             'output_r': instance['r'],
#             'head': head,
#             'tail': tail,
#             "sentence": sent,
#             'input_mask': -1, 
#         }
        
#         mask_positions = (input_ids == mask_token_id_to_find).nonzero().flatten()
#         if len(mask_positions) > 0:
#             current_row_data['input_mask'] = mask_positions[0].item()

#         if emb in [2, 3, 4]:
#             current_row_data['input_e1_pos'] = -1
#             current_row_data['input_e2_pos'] = -1
#             if e1_pos_token_id is not None:
#                 e1_marker_positions = (input_ids == e1_pos_token_id).nonzero().flatten()
#                 if len(e1_marker_positions) > 0:
#                     current_row_data['input_e1_pos'] = e1_marker_positions[0].item()
#             if e2_pos_token_id is not None: 
#                 e2_marker_positions = (input_ids == e2_pos_token_id).nonzero().flatten()
#                 if len(e2_marker_positions) > 0:
#                     current_row_data['input_e2_pos'] = e2_marker_positions[0].item()
        
#         rows.append(current_row_data)

#     if not rows:
#         print("Warning: No data rows were processed.")
#         return pd.DataFrame()

#     complete_fewrel = pd.DataFrame(rows)
#     if complete_fewrel.empty:
#          print("Warning: DataFrame is empty after processing rows.")
#          return complete_fewrel

#     complete_fewrel['output_label'] = pd.factorize(complete_fewrel['output_r'])[0]

#     tokens_tensor = torch.stack(complete_fewrel['input_tokens'].tolist(), dim=0)
#     attention_mask_tensor = torch.stack(complete_fewrel['input_attention_mask'].tolist(), dim=0)

#     dataset_tensors = [tokens_tensor, attention_mask_tensor]
#     if emb == 1 or emb == 5:
#         dataset_tensors.append(torch.Tensor(complete_fewrel['input_mask'].tolist()).long())
#     elif emb == 2:
#         dataset_tensors.append(torch.Tensor(complete_fewrel['input_e1_pos'].tolist()).long())
#     elif emb == 3:
#         dataset_tensors.append(torch.Tensor(complete_fewrel['input_mask'].tolist()).long()) 
#         dataset_tensors.append(torch.Tensor(complete_fewrel['input_e2_pos'].tolist()).long())
#     elif emb == 4:
#         dataset_tensors.append(torch.Tensor(complete_fewrel['input_mask'].tolist()).long())
#         dataset_tensors.append(torch.Tensor(complete_fewrel['input_e1_pos'].tolist()).long())
#         dataset_tensors.append(torch.Tensor(complete_fewrel['input_e2_pos'].tolist()).long())
    
#     dataset = TensorDataset(*dataset_tensors)
#     dataloader = DataLoader(dataset, num_workers=1, batch_size=24, shuffle=False)


#     bert_for_embeddings.to(device)
#     bert_for_embeddings.eval()
    
#     embeddings_list = [] 

#     if emb == 1:
#         with torch.no_grad():
#             batch_embeddings_list = []
#             for batch in tqdm(dataloader, desc="Embedding (emb=1)"):
#                 tokens_b, attention_mask_b, mask_b = batch 
#                 tokens_b, attention_mask_b, mask_b = tokens_b.to(device), attention_mask_b.to(device), mask_b.to(device)
                
#                 valid_indices = mask_b != -1
#                 if not torch.any(valid_indices): continue
                
#                 model_output = bert_for_embeddings(input_ids=tokens_b[valid_indices], attention_mask=attention_mask_b[valid_indices])
#                 out = model_output[0] 
#                 arange = torch.arange(out.shape[0], device=device)
#                 embedding = out[arange, mask_b[valid_indices]].detach()
#                 batch_embeddings_list.append(embedding)
#             if batch_embeddings_list:
#                 embeddings_list = list(torch.cat(batch_embeddings_list, dim=0).cpu())
#             with open("embeddings_mask_only_sms.pkl", "wb") as f: pickle.dump(embeddings_list[:10], f)

#     elif emb == 2:
#         with torch.no_grad():
#             batch_embeddings_list = []
#             for batch in tqdm(dataloader, desc="Embedding (emb=2)"):
#                 tokens_b, attention_mask_b, e1_b = batch 
#                 tokens_b, attention_mask_b, e1_b = tokens_b.to(device), attention_mask_b.to(device), e1_b.to(device)

#                 valid_indices = e1_b != -1
#                 if not torch.any(valid_indices): continue
                
#                 # Prepare input dictionary for RelationEncoder model
#                 input_dict = {
#                     "input_ids": tokens_b[valid_indices],
#                     "attention_mask": attention_mask_b[valid_indices]
#                 }
#                 out = bert_for_embeddings(input_dict) ## CORRECTED CALL ##
#                 # 'out' is directly the hidden states tensor as per README: (batch, seq_len, hidden_dim)
                
#                 arange = torch.arange(out.shape[0], device=device)
#                 embedding = out[arange, e1_b[valid_indices]].detach()
#                 batch_embeddings_list.append(embedding)
#             if batch_embeddings_list:
#                 embeddings_list = list(torch.cat(batch_embeddings_list, dim=0).cpu())
#             with open("embeddings_e1_only_sms.pkl", "wb") as f: pickle.dump(embeddings_list[:10], f)

#     elif emb == 3:
#         with torch.no_grad():
#             batch_embeddings_list = []
#             for batch in tqdm(dataloader, desc="Embedding (emb=3)"):
#                 tokens_b, attention_mask_b, _, e2_b = batch 
#                 tokens_b, attention_mask_b, e2_b = tokens_b.to(device), attention_mask_b.to(device), e2_b.to(device)

#                 valid_indices = e2_b != -1
#                 if not torch.any(valid_indices): continue

#                 input_dict = {
#                     "input_ids": tokens_b[valid_indices],
#                     "attention_mask": attention_mask_b[valid_indices]
#                 }
#                 out = bert_for_embeddings(input_dict) ## CORRECTED CALL ##
                
#                 arange = torch.arange(out.shape[0], device=device)
#                 embedding = out[arange, e2_b[valid_indices]].detach()
#                 batch_embeddings_list.append(embedding)
#             if batch_embeddings_list:
#                 embeddings_list = list(torch.cat(batch_embeddings_list, dim=0).cpu())
#             with open("embeddings_e2_only_sms.pkl", "wb") as f: pickle.dump(embeddings_list[:10], f)

#     elif emb == 4:
#         with torch.no_grad():
#             batch_embeddings_list = []
#             for batch in tqdm(dataloader, desc="Embedding (emb=4)"):
#                 tokens_b, attention_mask_b, mask_b, e1_b, e2_b = batch
#                 tokens_b, attention_mask_b, mask_b, e1_b, e2_b = \
#                     tokens_b.to(device), attention_mask_b.to(device), mask_b.to(device), e1_b.to(device), e2_b.to(device)

#                 valid_indices = (mask_b != -1) & (e1_b != -1) & (e2_b != -1)
#                 if not torch.any(valid_indices): continue
                
#                 input_dict = {
#                     "input_ids": tokens_b[valid_indices],
#                     "attention_mask": attention_mask_b[valid_indices]
#                 }
#                 out = bert_for_embeddings(input_dict) ## CORRECTED CALL ##
                
#                 arange = torch.arange(out.shape[0], device=device)
#                 e1_emb = out[arange, e1_b[valid_indices]]
#                 e2_emb = out[arange, e2_b[valid_indices]]
#                 mask_emb = out[arange, mask_b[valid_indices]]
#                 combined = torch.cat([e1_emb, e2_emb, mask_emb], dim=1).detach()
#                 batch_embeddings_list.append(combined)
#             if batch_embeddings_list:
#                 embeddings_list = list(torch.cat(batch_embeddings_list, dim=0).cpu())
#             with open("embeddings_e1_e2_mask_concat_sms.pkl", "wb") as f: pickle.dump(embeddings_list[:10], f)
    
#     elif emb == 5: 
#         TOP_N = 1
#         with torch.no_grad():
#             batch_embeddings_list = []
#             for batch in tqdm(dataloader, desc="Embedding (emb=5)"):
#                 tokens_b, attention_mask_b, mask_b = batch
#                 tokens_b, attention_mask_b, mask_b = tokens_b.to(device), attention_mask_b.to(device), mask_b.to(device)

#                 valid_indices = mask_b != -1
#                 if not torch.any(valid_indices): continue
                
#                 tokens_vf, attention_mask_vf, mask_vf = \
#                     tokens_b[valid_indices], attention_mask_b[valid_indices], mask_b[valid_indices]

#                 output_lm = bert_for_embeddings(input_ids=tokens_vf, attention_mask=attention_mask_vf) 
#                 logits = output_lm.logits 
#                 vocab_probs = torch.softmax(logits, dim=-1) 

#                 current_valid_batch_size = tokens_vf.size(0)
#                 top_embeddings_for_batch = []

#                 for i in range(current_valid_batch_size):
#                     mask_pos_in_instance = mask_vf[i].item()
#                     top_n_token_ids = torch.topk(vocab_probs[i, mask_pos_in_instance], k=TOP_N).indices

#                     concat_embedding_for_instance = []
#                     for token_id_replace in top_n_token_ids:
#                         modified_input_ids_instance = tokens_vf[i].clone()
#                         modified_input_ids_instance[mask_pos_in_instance] = token_id_replace
                        
#                         out_mod_all_layers = bert_for_embeddings(
#                             input_ids=modified_input_ids_instance.unsqueeze(0), 
#                             attention_mask=attention_mask_vf[i].unsqueeze(0)
#                         ) # output_hidden_states=True is set at model init for emb=5
                        
#                         last_hidden_state = out_mod_all_layers.hidden_states[-1]
#                         concat_embedding_for_instance.append(last_hidden_state[0, mask_pos_in_instance])
                    
#                     final_embed_instance = torch.cat(concat_embedding_for_instance, dim=-1).detach()
#                     top_embeddings_for_batch.append(final_embed_instance)
                
#                 if top_embeddings_for_batch:
#                      batch_embeddings_list.append(torch.stack(top_embeddings_for_batch, dim=0))
            
#             if batch_embeddings_list:
#                 embeddings_list = list(torch.cat(batch_embeddings_list, dim=0).cpu())
#             with open("embeddings_topk1_sms.pkl", "wb") as f: pickle.dump(embeddings_list[:10], f)

#     bert_for_embeddings.to('cpu')
    
#     # Assign embeddings back to DataFrame
#     if embeddings_list:
#         valid_rows_mask = pd.Series([True] * len(complete_fewrel), index=complete_fewrel.index)
#         if emb == 1 or emb == 5: valid_rows_mask &= (complete_fewrel['input_mask'] != -1)
#         elif emb == 2: valid_rows_mask &= (complete_fewrel['input_e1_pos'] != -1)
#         elif emb == 3: valid_rows_mask &= (complete_fewrel['input_e2_pos'] != -1)
#         elif emb == 4: valid_rows_mask &= ((complete_fewrel['input_mask'] != -1) & \
#                                            (complete_fewrel['input_e1_pos'] != -1) & \
#                                            (complete_fewrel['input_e2_pos'] != -1))
        
#         target_df_for_embeddings = complete_fewrel[valid_rows_mask].copy()

#         if len(embeddings_list) == len(target_df_for_embeddings):
#             target_df_for_embeddings['embedding'] = embeddings_list
#             print(f"Successfully generated and assigned {len(embeddings_list)} embeddings.")
#             return target_df_for_embeddings
#         elif embeddings_list: 
#              print(f"Warning: Mismatch assigning embeddings. Generated: {len(embeddings_list)}, Target Rows: {len(target_df_for_embeddings)}.")
#              if len(embeddings_list) <= len(target_df_for_embeddings) and len(embeddings_list) > 0 :
#                  print(f"Assigning {len(embeddings_list)} embeddings to the first {len(embeddings_list)} identified valid rows.")
#                  target_df_for_embeddings = target_df_for_embeddings.iloc[:len(embeddings_list)].copy()
#                  target_df_for_embeddings['embedding'] = embeddings_list
#                  return target_df_for_embeddings
#              else: 
#                  print("Returning target_df without embeddings due to unresolved mismatch or empty embeddings list.")
#                  return target_df_for_embeddings # Or complete_fewrel if preferred
#         else: 
#             print("Warning: `embeddings_list` is empty, though some rows were identified as valid. No embeddings assigned.")
#             return target_df_for_embeddings
#     else: 
#         print("Warning: No embeddings were generated. Returning original processed DataFrame without 'embedding' column.")
#         return complete_fewrel

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
