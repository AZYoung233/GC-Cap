import argparse
import requests
import json
import time
import random
import hmac
import hashlib
import re
import multiprocessing
from multiprocessing import Lock
import csv
import os
import ast
import networkx as nx
import pandas as pd
import sys
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import numpy as np
import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from prompt_content import ENTITY_RELATIONSHIPS_GENERATION_PROMPT_V4


# Define all fieldnames used in the CSV
ALL_FIELDNAMES = [
    'image_id', 'img_name', 'gt_caption', 'vlm_caption',
    'gt_entities', 'vlm_entities',
    'precision_analysis', 'recall_analysis',
    'precision_score', 'recall_score', 'f1_score'
]

# Fieldnames for generate phase only
GENERATE_FIELDNAMES = [
    'image_id', 'img_name', 'gt_caption', 'vlm_caption',
    'gt_entities', 'vlm_entities'
]

# Fieldnames added in compute phase
COMPUTE_FIELDNAMES = [
    'precision_analysis', 'recall_analysis',
    'precision_score', 'recall_score', 'f1_score'
]


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across different libraries
    
    Args:
        seed (int): Random seed value, default is 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class API_Call:
    def __init__(self, api_model_name):
        self.api_model_name = api_model_name
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
    def __call__(self, messages):
        completion = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        result = completion.choices[0].message.content
        return json.loads(result)


class API_Call_Azure:
    def __init__(self, api_model_name):
        self.client = AzureOpenAI(
            api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
            api_key=os.environ.get("AZURE_API_KEY", "YOUR_API_KEY_HERE"),
            azure_endpoint=os.environ.get("AZURE_ENDPOINT", "https://your-endpoint.openai.azure.com"),
        )
        self.api_model_name = api_model_name

    def __call__(self, messages):
        completion = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        result = completion.choices[0].message.content
        return json.loads(result)


class Graph_Utils:
    def __init__(self, api_model_name, api_backend='azure'):
        """
        Initialize Graph_Utils with specified API backend
        
        Args:
            api_model_name: Model name/ID to use
        api_backend: API backend to use, one of 'azure', 'openai'
        """
        self.api_backend = api_backend.lower()
        
        if self.api_backend == 'azure':
            self.gpt_instance = API_Call_Azure(api_model_name)
        elif self.api_backend == 'openai':
            self.gpt_instance = API_Call(api_model_name)
        else:
            raise ValueError(f"Unknown API backend: {api_backend}. Supported: 'azure', 'openai'")

    def validate_triples_list(self, triples_list):
        """
        Validate triples list by attempting to unpack each triple as (s, p, o)
        """
        try:
            for triple in triples_list:
                s, p, o = triple
            return True
        except:
            return False
        

    def generate_triples_list(self, input_text):
        input_text = "The game character's caption is: [" + input_text + "]."

        messages = [
            {"role": "system", "content": ENTITY_RELATIONSHIPS_GENERATION_PROMPT_V4},
            {"role": "user", "content": input_text},
        ]
        
        max_retries = 5
        triples_list = None
        
        for attempt in range(max_retries):
            try:
                response = self.gpt_instance(messages)
                
                if response is None or not isinstance(response, dict):
                    print(f"Attempt {attempt + 1}: Received empty or non-dict response")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        break
                else:
                    triples_list = response.get("triples", [])
                    
                    if not triples_list or len(triples_list) == 0:
                        print(f"Attempt {attempt + 1}: Triples list is empty")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            break
                    
                    # Validate all triples can be unpacked as (s, p, o)
                    if not self.validate_triples_list(triples_list):
                        print(f"Attempt {attempt + 1}: Triples validation failed, regenerating...")
                        triples_list = None
                        if attempt < max_retries - 1:
                            continue
                        else:
                            break
                    
                    # All validations passed
                    break
                        
            except Exception as e:
                print(f"Attempt {attempt + 1}: Unexpected error: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    break
        
        if triples_list is None:
            print("Failed to extract valid triples after 5 attempts. Terminating program.")
            sys.exit(1)
        
        return triples_list

    def generate_graph(self, triples_list):
        Graph = nx.DiGraph()
        for s, p, o in triples_list:
            Graph.add_edge(s, o, label=p)
        Filter_entities_list = self.get_graph_node_with_neighbor(Graph)
        return Filter_entities_list, Graph, triples_list

    def get_graph_node_with_neighbor(self, graph):
        return [node for node in graph.nodes if graph.out_degree(node) > 0]

    def describe_entity_with_edges(self, graph, entity_name):
        if entity_name not in graph:
            return f"{entity_name} has no outgoing edges"

        out_edges = list(graph.out_edges(entity_name, data=True))
        if not out_edges:
            return f"{entity_name} has no outgoing edges"

        parts = []
        for u, v, data in out_edges:
            label = data.get('label', 'related to')
            parts.append(f"{u} {label} {v}.")

        return ''.join(parts)


class Text_model:
    def __init__(self, embedding_model_name, rerank_model_name, device):
        self.device = device
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name, attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map=self.device)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, padding_side="left")
        self.rerank_model = AutoModelForCausalLM.from_pretrained(rerank_model_name, attn_implementation="flash_attention_2", torch_dtype=torch.float16, device_map=self.device)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name, padding_side="left")
        self.max_length = 8096

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_entities_similarity_matrix(self, Filter_entities_list_gt, Filter_entities_list_vlm):
        input_texts = Filter_entities_list_vlm + Filter_entities_list_gt

        batch_dict = self.embedding_tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = (embeddings[:len(Filter_entities_list_vlm)] @ embeddings[len(Filter_entities_list_vlm):].T)

        return scores

    def format_instruction(self, query, doc):
        output = "<SentenceA>: {query}\n<SentenceB>: {doc}".format(query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        inputs = self.rerank_tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.rerank_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.rerank_model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.rerank_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def get_description_similarity(self, queries, documents, prefix_system):
        prefix = prefix_system + "<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.rerank_tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.rerank_tokenizer.encode(suffix, add_special_tokens=False)
        self.token_false_id = self.rerank_tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.rerank_tokenizer.convert_tokens_to_ids("yes")

        pairs = [self.format_instruction(query, doc) for query, doc in zip([queries], [documents])]
        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        return scores[0]


class Graph_Comparison_F1:
    def __init__(self, text_model, graph_utils):
        self.text_model = text_model
        self.graph_utils = graph_utils

        self.precision_prefix_system = """
                <|im_start|>system
                You are a Logical Verifier. Your task is to determine if Sentence A (claim) is a strict informational subset of Sentence B (facts).

                ### Core Principle: Zero Addition

                Sentence A **must not** contain any information that is not explicitly in Sentence B.

                ### Judgment Rules:

                You will judge as **`no (not supported)`** if **ANY** of these rules are broken:

                1.  **No Added Details**: A introduces any new attribute.
                    -   B: "a car" -> A: "a **red** car" = **`no`**

                2.  **No Added Specificity**: A uses a more specific term than B.
                    -   B: "a blue shirt" -> A: "a **dark blue** shirt" = **`no`**
                    -   B: "a metal box" -> A: "a **steel** box" = **`no`**

                3.  **No Contradictions**: Any part of A conflicts with B.

                4.  **No Missing Support**: Any part of A is not mentioned in B.

                You will judge as **`yes (supported)`** **only if** A perfectly passes all rules. Generalizations are acceptable (e.g., B: "pistol" -> A: "gun").

                **Output Requirements**:  
                Note that you can only output \"yes\" or \"no\".
                <|im_end|>
            """

        self.recall_prefix_system = """
                <|im_start|>system
                You are an information coverage auditor. Your task is to determine whether the "submitted content" (Sentence B) fully covers all information points in the "requirement list" (Sentence A, ground truth).  

                **Core Rules**:  
                1. **Coverage Requirement**: B must include all attributes and values from A (type, color, material, shape, location, etc.). Equivalent expressions are allowed (e.g., "silver-white long hair" ≈ "long silver-white hair").  
                2. **Tolerance for Additional Information**: Even if B contains extra information not declared in A, it does not affect the judgment (these redundant details are handled in other evaluation steps).  
                3. **Conflict Rejection**: If B contradicts A in any attribute, it is judged as "incomplete/inconsistent."  
                4. **Non-informative Words Ignored**: Modal particles, stop words, and order differences do not affect the judgment.  

                **Judgment Criteria**:  
                - **Yes (Full Coverage)**: B covers all information points in A (additional information is allowed as long as there is no conflict).  
                - **No (Incomplete/Inconsistent)**: Any information point in A is missing in B, or a conflict exists.

                **Output Requirements**:  
                Note that you can only output \"yes\" or \"no\".
                <|im_end|>
                    """

    def precision_compute(self, filter_entities_list_gt, filter_entities_list_vlm, Entities_similarity_matrix, Graph_gt, Graph_vlm):
        precision_score = 0
        precision_analysis = dict()

        for i, row in enumerate(Entities_similarity_matrix):
            j = torch.argmax(row).item()
            max_similarity = row[j].item()

            vlm_entity_name = filter_entities_list_vlm[i]
            gt_entity_name = filter_entities_list_gt[j]

            vlm_entity_text = self.graph_utils.describe_entity_with_edges(Graph_vlm, vlm_entity_name)
            gt_entity_text = self.graph_utils.describe_entity_with_edges(Graph_gt, gt_entity_name)

            describition_similarity = self.text_model.get_description_similarity(vlm_entity_text, gt_entity_text, self.precision_prefix_system)
            final_score = max_similarity * describition_similarity

            precision_score += final_score
            analysis_result = {
                "most_similar_gt_entity_name": gt_entity_name,
                "max_similarity": round(max_similarity, 3),
                "vlm_entity_text": vlm_entity_text,
                "gt_entity_text": gt_entity_text,
                "describition_similarity": round(describition_similarity, 3),
                "final_score": round(final_score, 3),
            }
            precision_analysis[vlm_entity_name] = analysis_result
        
        precision_score /= len(filter_entities_list_vlm)
        return round(precision_score, 3), precision_analysis

    def recall_compute(self, filter_entities_list_gt, filter_entities_list_vlm, Entities_similarity_matrix, Graph_gt, Graph_vlm):
        recall_score = 0
        recall_analysis = dict()

        for j, row in enumerate(Entities_similarity_matrix):
            i = torch.argmax(row).item()
            max_similarity = row[i].item()

            gt_entity_name = filter_entities_list_gt[j]
            vlm_entity_name = filter_entities_list_vlm[i]

            gt_entity_text = self.graph_utils.describe_entity_with_edges(Graph_gt, gt_entity_name)
            vlm_entity_text = self.graph_utils.describe_entity_with_edges(Graph_vlm, vlm_entity_name)

            describition_similarity = self.text_model.get_description_similarity(gt_entity_text, vlm_entity_text, self.recall_prefix_system)
            final_score = max_similarity * describition_similarity

            recall_score += final_score

            analysis_result = {
                "most_similar_vlm_entity_name": vlm_entity_name,
                "max_similarity": round(max_similarity, 3),
                "gt_entity_text": gt_entity_text,
                "vlm_entity_text": vlm_entity_text,
                "describition_similarity": round(describition_similarity, 3),
                "final_score": round(final_score, 3),
            }
            recall_analysis[gt_entity_name] = analysis_result
        
        recall_score /= len(filter_entities_list_gt)
        return round(recall_score, 3), recall_analysis
    
    def f1_compute(self, precision_score, recall_score):
        if precision_score + recall_score > 0:
            f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score)
        else:
            f1 = 0
        return round(f1, 3)


# ==================== CSV Helper Functions ====================

def load_existing_data(save_path):
    """
    Load existing data from CSV file as a dict keyed by img_name
    Returns: dict {img_name: row_dict}
    """
    existing_data = {}
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_data[row['img_name']] = row
    return existing_data


def save_all_data(save_path, data_dict):
    """
    Save all data to CSV file, sorted by image_id
    Args:
        save_path: path to save
        data_dict: dict {img_name: row_dict}
    """
    if not data_dict:
        return
    
    # Convert to list and sort by image_id
    data_list = list(data_dict.values())
    data_list.sort(key=lambda x: int(x.get('image_id', 0)))
    
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDNAMES)
        writer.writeheader()
        for row in data_list:
            # Fill missing fields with empty string
            complete_row = {k: row.get(k, '') for k in ALL_FIELDNAMES}
            writer.writerow(complete_row)


def get_processed_status(existing_data, phase='generate'):
    """
    Get sets of processed img_names for different phases
    Args:
        existing_data: dict from load_existing_data
        phase: 'generate' or 'compute'
    Returns:
        set of img_names that have been processed for that phase
    """
    processed = set()
    for img_name, row in existing_data.items():
        if phase == 'generate':
            # Check if vlm_entities field exists and is not empty
            if row.get('vlm_entities', '').strip():
                processed.add(img_name)
        elif phase == 'compute':
            # Check if f1_score field exists and is not empty
            if row.get('f1_score', '').strip():
                processed.add(img_name)
    return processed


# ==================== Generate Mode Functions ====================

def process_single_image_generate(args_tuple):
    """
    Generate mode: Only generate VLM entities triples list
    Returns: dict with generate phase fields
    """
    image_id, img_name, gt_caption, gt_entities, vlm_caption, graph_utils = args_tuple
    
    single_result = dict()
    single_result['image_id'] = image_id
    single_result['img_name'] = img_name
    single_result['gt_caption'] = gt_caption
    single_result['vlm_caption'] = vlm_caption
    
    # Store gt_entities as string
    single_result['gt_entities'] = '\n'.join(
        str(item) for item in gt_entities
    )
    
    # Generate VLM entities
    original_entities_list_vlm = graph_utils.generate_triples_list(vlm_caption)
    single_result['vlm_entities'] = '\n'.join(
        str(item) for item in original_entities_list_vlm
    )
    
    print(f"[Generate] Image {img_name} processed successfully")
    return img_name, single_result


# ==================== Compute Mode Functions ====================

def process_single_image_compute(args_tuple):
    """
    Compute mode: Compute scores based on pre-generated entities
    Returns: dict with compute phase fields to update
    """
    image_id, img_name, gt_caption, gt_entities, vlm_caption, vlm_entities, text_model, graph_utils, graph_comparison_f1 = args_tuple
    
    update_fields = dict()
    
    # Process GT graph
    filter_entities_list_gt, Graph_gt, original_entities_list_gt = graph_utils.generate_graph(gt_entities)
    
    # Process VLM graph (entities already generated)
    filter_entities_list_vlm, Graph_vlm, original_entities_list_vlm = graph_utils.generate_graph(vlm_entities)
    
    Entities_similarity_matrix = text_model.get_entities_similarity_matrix(filter_entities_list_gt, filter_entities_list_vlm)
    
    precision_score, precision_analysis = graph_comparison_f1.precision_compute(
        filter_entities_list_gt, filter_entities_list_vlm, Entities_similarity_matrix, Graph_gt, Graph_vlm
    )
    recall_score, recall_analysis = graph_comparison_f1.recall_compute(
        filter_entities_list_gt, filter_entities_list_vlm, Entities_similarity_matrix.T, Graph_gt, Graph_vlm
    )
    
    f1_score = graph_comparison_f1.f1_compute(precision_score, recall_score)
    
    update_fields['precision_analysis'] = '\n\n'.join(
        f"{k}: {v}" for k, v in precision_analysis.items()
    )
    update_fields['recall_analysis'] = '\n\n'.join(
        f"{k}: {v}" for k, v in recall_analysis.items()
    )
    
    update_fields['precision_score'] = precision_score
    update_fields['recall_score'] = recall_score
    update_fields['f1_score'] = f1_score
    
    print(f"[Compute] Image {img_name} processed successfully")
    return img_name, update_fields


def compute_and_print_average_metrics(save_path):
    """
    Read results CSV file, compute and print average precision, recall, F1 scores,
    and average word count of VLM captions
    """
    df = pd.read_csv(save_path)
    
    # Filter rows that have f1_score
    df_computed = df[df['f1_score'].notna() & (df['f1_score'] != '')]
    
    if len(df_computed) == 0:
        print("No computed results found.")
        return
    
    avg_precision = df_computed['precision_score'].astype(float).mean()
    avg_recall = df_computed['recall_score'].astype(float).mean()
    avg_f1 = df_computed['f1_score'].astype(float).mean()
    
    # Calculate average word count for VLM captions
    vlm_captions = df_computed['vlm_caption'].dropna()
    if len(vlm_captions) > 0:
        word_counts = vlm_captions.apply(lambda x: len(str(x).strip().split()))
        avg_word_count = word_counts.mean()
        caption_count = len(vlm_captions)
    else:
        avg_word_count = 0
        caption_count = 0
    
    print("\n" + "="*50)
    print("Evaluation Results Summary")
    print("="*50)
    print(f"Total samples with scores: {len(df_computed)}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"VLM Caption Average Word Count: {avg_word_count:.2f} (over {caption_count} captions)")
    print("="*50)


def run_generate_mode(args):
    """
    Generate mode: Only generate VLM entities triples list using API
    """
    print("="*50)
    print("Running GENERATE phase")
    print("="*50)
    
    save_path = args.save_path
    cap_file_gt = args.cap_file_gt
    cap_file_model = args.cap_file_model
    if cap_file_model is None:
        raise ValueError("--cap_file_model is required for 'generate' and 'combine' modes")
    
    # Read GT annotations
    caption_annotations = []
    with open(cap_file_gt, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            caption_annotations.append({
                'image_id': row['image_id'],
                'img_name': row['img_name'].replace('\\', '/'),
                'caption': row['Human Label'],
                'gt_entities': row['gt_entities']
            })
    
    # Read VLM results
    caption_results = {}
    with open(cap_file_model, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            img_name = row['img_name'].replace('\\', '/')
            caption = row['caption']
            caption_results[img_name] = caption
    
    caption_annotations_dict = {}
    for item in caption_annotations:
        image_id = item['image_id']
        caption = item['caption']
        img_name = item['img_name']
        gt_entities_string = item['gt_entities']
        gt_entities = [ast.literal_eval(line.strip()) for line in gt_entities_string.splitlines() if line.strip()]
        caption_annotations_dict[img_name] = {'caption': caption, 'image_id': image_id, 'gt_entities': gt_entities}
    
    # Load existing data and get processed status
    existing_data = load_existing_data(save_path)
    processed_generate = get_processed_status(existing_data, phase='generate')
    
    # Initialize graph_utils with specified backend
    graph_utils = Graph_Utils(
        args.api_model_name, 
        api_backend=args.api_backend
    )
    
    img_names = list(caption_annotations_dict.keys())
    filter_img_names = [id for id in img_names if id not in processed_generate]
    
    print(f"Total images: {len(img_names)}, Already generated: {len(processed_generate)}, To generate: {len(filter_img_names)}")
    
    if len(filter_img_names) == 0:
        print("All images already processed in generate phase.")
        return
    
    args_list = []
    for img_name in filter_img_names:
        gt_caption = caption_annotations_dict[img_name]['caption']
        gt_entities = caption_annotations_dict[img_name]['gt_entities']
        original_idx = caption_annotations_dict[img_name]['image_id']
        vlm_caption = caption_results.get(img_name, "")
        args_list.append((original_idx, img_name, gt_caption, gt_entities, vlm_caption, graph_utils))
    
    # Process with thread pool and collect results
    results_lock = threading.Lock()
    
    def process_and_update(args_item):
        img_name, result = process_single_image_generate(args_item)
        with results_lock:
            existing_data[img_name] = result
            # Save periodically (every result for safety)
            save_all_data(save_path, existing_data)
        return result
    
    with ThreadPoolExecutor(max_workers=args.generate_max_workers) as executor:
        list(tqdm(
            executor.map(process_and_update, args_list),
            total=len(args_list),
            desc="Generating entities"
        ))
    
    # Final save
    save_all_data(save_path, existing_data)
    print(f"\nGenerate phase completed. Results saved to: {save_path}")


def run_compute_mode(args):
    """
    Compute mode: Compute scores based on pre-generated entities
    """
    print("="*50)
    print("Running COMPUTE phase")
    print("="*50)
    
    save_path = args.save_path
    
    # Load existing data
    existing_data = load_existing_data(save_path)
    
    if not existing_data:
        print(f"Error: No data found in {save_path}")
        print("Please run generate mode first.")
        sys.exit(1)
    
    # Get processed status
    processed_generate = get_processed_status(existing_data, phase='generate')
    processed_compute = get_processed_status(existing_data, phase='compute')
    
    # Find images that have been generated but not computed
    to_compute = processed_generate - processed_compute
    
    print(f"Total images with entities: {len(processed_generate)}, Already computed: {len(processed_compute)}, To compute: {len(to_compute)}")
    
    if len(to_compute) == 0:
        print("All images already processed in compute phase.")
        compute_and_print_average_metrics(save_path)
        return
    
    # Initialize models
    text_model = Text_model(args.embedding_model_name, args.rerank_model_name, args.device)
    graph_utils = Graph_Utils(
        args.api_model_name,
        api_backend=args.api_backend
    )
    graph_comparison_f1 = Graph_Comparison_F1(text_model=text_model, graph_utils=graph_utils)
    
    args_list = []
    for img_name in to_compute:
        row = existing_data[img_name]
        gt_entities = [ast.literal_eval(line.strip()) for line in row['gt_entities'].splitlines() if line.strip()]
        vlm_entities = [ast.literal_eval(line.strip()) for line in row['vlm_entities'].splitlines() if line.strip()]
        args_list.append((
            row['image_id'], img_name, row['gt_caption'], gt_entities,
            row['vlm_caption'], vlm_entities,
            text_model, graph_utils, graph_comparison_f1
        ))
    
    # Process with thread pool and collect results
    results_lock = threading.Lock()
    
    def process_and_update(args_item):
        img_name, update_fields = process_single_image_compute(args_item)
        with results_lock:
            existing_data[img_name].update(update_fields)
            # Save periodically
            save_all_data(save_path, existing_data)
        return update_fields
    
    with ThreadPoolExecutor(max_workers=args.compute_max_workers) as executor:
        list(tqdm(
            executor.map(process_and_update, args_list),
            total=len(args_list),
            desc="Computing scores"
        ))
    
    # Final save
    save_all_data(save_path, existing_data)
    compute_and_print_average_metrics(save_path)
    print("\nCompute phase completed.")


def run_combine_mode(args):
    """
    Combine mode: Execute generate phase first, then execute compute phase
    """
    print("="*50)
    print("Running in COMBINE mode (Generate -> Compute pipeline)")
    print("="*50)
    print(f"\nOutput file: {args.save_path}")
    
    # Phase 1: Generate
    print("\n" + "#"*50)
    print("# Phase 1: Generate VLM entities")
    print("#"*50)
    run_generate_mode(args)
    
    # Phase 2: Compute
    print("\n" + "#"*50)
    print("# Phase 2: Compute scores")
    print("#"*50)
    run_compute_mode(args)
    
    print("\n" + "="*50)
    print("COMBINE mode completed!")
    print("="*50)


def main(args):
    set_random_seed(args.seed)
    
    print("="*50)
    print("Start evaluation")
    print("="*50)
    print("Arguments:", args)
    print("="*50)
    
    if args.mode == 'generate':
        run_generate_mode(args)
    elif args.mode == 'compute':
        run_compute_mode(args)
    elif args.mode == 'combine':
        run_combine_mode(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'. Please use 'generate', 'compute', or 'combine'.")
        sys.exit(1)
    
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default='combine',
        choices=['generate', 'compute', 'combine'],
        help="Evaluation mode: 'generate' (only generate VLM entities), 'compute' (compute scores from pre-generated entities), 'combine' (generate then compute in sequence)"
    )
    
    # Input files
    parser.add_argument(
        "--cap_file_gt",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GC-Bench',
                             'GC-Benchmark_1009_186_with_gt_entities_reorganized.csv'),
        help="Ground truth caption file"
    )
    parser.add_argument(
        "--cap_file_model",
        type=str,
        default=None,
        help="VLM caption results file (required for 'generate' and 'combine' modes)"
    )

    # Single output file for all modes
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save evaluation results (used for all modes, supports incremental updates)"
    )
    
    # Model configurations
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model name"
    )
    parser.add_argument(
        "--rerank_model_name",
        type=str,
        default="Qwen/Qwen3-Reranker-4B",
        help="Rerank model name"
    )
    parser.add_argument(
        "--api_model_name",
        type=str,
        default="deepseek-v3.2",
        help="API model name (e.g., 'gpt-5' for Azure/OpenAI)"
    )
    parser.add_argument(
        "--api_backend",
        type=str,
        default="openai",
        choices=['azure', 'openai'],
        help="API backend to use: 'azure' (Azure OpenAI), 'openai' (OpenAI compatible API)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for model"
    )
    
    # Processing configurations
    parser.add_argument(
        "--generate_max_workers",
        type=int,
        default=32,
        help="Max workers for parallel processing in generate phase (can be high since it's API bound)"
    )
    parser.add_argument(
        "--compute_max_workers",
        type=int,
        default=1,
        help="Max workers for parallel processing in compute phase (usually 1 since it's GPU bound)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    main(args)