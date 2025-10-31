"""
Fisher Information Matrix Calculator for Language Models
=========================================================

Computes diagonal Fisher Information Matrix using the empirical Fisher method.
Based on best practices from:
- mmatena/model_merging repository (Fisher calculation logic)
- Our previous fim_fixed.py (adapted for causal LM)

Usage:
    python fisher.py --model_path <path> --calibration_data <path> --device cuda:0

Output:
    CSV table with per-layer Fisher statistics:
    - Layer name
    - Total Fisher (sum of all Fisher values)
    - Percentage (relative to whole model)
    - Mean, Std, Max, Min Fisher values
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import argparse
import os
import pandas as pd
from collections import OrderedDict
import numpy as np


class TextDataset(Dataset):
    """Dataset for calibration texts"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }


def load_calibration_data(data_path, num_samples=None):
    """
    Load calibration data from file.
    
    Args:
        data_path: Path to JSON, JSONL, or TXT file
        num_samples: Maximum number of samples to load
    
    Returns:
        List of text strings
    """
    print(f"Loading calibration data from: {data_path}")
    
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = [item if isinstance(item, str) else item.get('text', '') for item in data]
            else:
                texts = [data.get('text', '')]
    
    elif data_path.endswith('.jsonl'):
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                texts.append(item if isinstance(item, str) else item.get('text', ''))
    
    elif data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    if num_samples:
        texts = texts[:num_samples]
    
    print(f"Loaded {len(texts)} calibration samples")
    return texts


def compute_fisher_information(model, tokenizer, calibration_texts, 
                                batch_size=4, max_length=512, 
                                device='cuda:0'):
    """
    Compute empirical Fisher Information Matrix (diagonal approximation).
    
    This uses the empirical Fisher approach:
        Fisher_ii = E[(∂ log p(y|x) / ∂θ_i)²]
    
    Where y is the actual observed output (next token in sequence).
    
    Args:
        model: HuggingFace causal language model
        tokenizer: Corresponding tokenizer
        calibration_texts: List of calibration strings
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to use (e.g., 'cuda:0')
    
    Returns:
        fisher_dict: {parameter_name: fisher_values_tensor}
    """
    print(f"\n{'='*70}")
    print(f"Computing Empirical Fisher Information Matrix")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Samples: {len(calibration_texts)}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    
    model.eval()
    
    # Prepare dataset
    dataset = TextDataset(calibration_texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Fisher accumulators (use float32 for numerical stability)
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param, dtype=torch.float32)
    
    # Loss function for per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(
        reduction='none',
        ignore_index=tokenizer.pad_token_id
    )
    
    total_samples = 0
    
    print(f"\nProcessing batches...")
    # Process batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Fisher")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_size_actual = input_ids.size(0)
        
        # Forward pass
        with torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            # Compute per-sample loss for causal LM
            # Shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Get per-token loss
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Reshape to [batch_size, seq_len]
            per_token_loss = per_token_loss.view(batch_size_actual, -1)
            
            # Count non-ignored tokens per sample
            non_ignored_tokens = (shift_labels != tokenizer.pad_token_id).sum(dim=1)
            non_ignored_tokens = torch.clamp(non_ignored_tokens, min=1)
            
            # Mean loss per sample
            per_sample_loss = per_token_loss.sum(dim=1) / non_ignored_tokens
            
            # Check for invalid losses
            if torch.isnan(per_sample_loss).any() or torch.isinf(per_sample_loss).any():
                print(f"Warning: Invalid loss in batch {batch_idx}, skipping")
                continue
            
            # Compute per-sample gradients and accumulate squared gradients
            for j in range(batch_size_actual):
                model.zero_grad()
                
                # Backward pass for sample j
                is_last_sample = (j == batch_size_actual - 1)
                per_sample_loss[j].backward(retain_graph=not is_last_sample)
                
                # Accumulate squared gradients (Fisher)
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Square the gradient and accumulate
                        fisher_dict[name] += param.grad.detach().pow(2).float()
                
                total_samples += 1
    
    # Normalize by number of samples
    print(f"\nNormalizing Fisher values over {total_samples} samples...")
    for name in fisher_dict:
        fisher_dict[name] /= total_samples
    
    print(f"✓ Fisher computation complete!")
    return fisher_dict


def aggregate_fisher_by_layer(fisher_dict):
    """
    Aggregate Fisher information by layer.
    
    Groups parameters by layer name and computes statistics.
    
    Args:
        fisher_dict: {parameter_name: fisher_tensor}
    
    Returns:
        DataFrame with columns: layer, total_fisher, percentage, mean, std, max, min
    """
    print(f"\n{'='*70}")
    print(f"Aggregating Fisher Information by Layer")
    print(f"{'='*70}")
    
    # Group by layer
    layer_stats = OrderedDict()
    
    for param_name, fisher_values in fisher_dict.items():
        # Extract layer name (everything before the last dot)
        # e.g., "model.layers.0.self_attn.q_proj.weight" -> "model.layers.0.self_attn.q_proj"
        layer_name = '.'.join(param_name.split('.')[:-1])
        
        # Convert to numpy for statistics
        fisher_np = fisher_values.cpu().numpy().flatten()
        
        if layer_name not in layer_stats:
            layer_stats[layer_name] = {
                'fisher_values': [],
                'param_names': []
            }
        
        layer_stats[layer_name]['fisher_values'].extend(fisher_np.tolist())
        layer_stats[layer_name]['param_names'].append(param_name)
    
    # Compute statistics for each layer
    results = []
    total_fisher_all = sum(sum(stats['fisher_values']) for stats in layer_stats.values())
    
    for layer_name, stats in layer_stats.items():
        fisher_vals = np.array(stats['fisher_values'])
        
        total_fisher = float(np.sum(fisher_vals))
        percentage = (total_fisher / total_fisher_all * 100) if total_fisher_all > 0 else 0
        mean_fisher = float(np.mean(fisher_vals))
        std_fisher = float(np.std(fisher_vals))
        max_fisher = float(np.max(fisher_vals))
        min_fisher = float(np.min(fisher_vals))
        num_params = len(fisher_vals)
        
        results.append({
            'layer': layer_name,
            'total_fisher': total_fisher,
            'percentage': percentage,
            'mean': mean_fisher,
            'std': std_fisher,
            'max': max_fisher,
            'min': min_fisher,
            'num_params': num_params
        })
    
    # Sort by total_fisher descending
    results.sort(key=lambda x: x['total_fisher'], reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print(f"✓ Aggregated {len(df)} layers")
    print(f"✓ Total Fisher (all layers): {total_fisher_all:.6e}")
    
    return df


def print_summary_table(df, top_n=20):
    """Print a nicely formatted summary table"""
    print(f"\n{'='*70}")
    print(f"Fisher Information Summary (Top {top_n} Layers)")
    print(f"{'='*70}\n")
    
    # Configure pandas display
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.6e}')
    
    print(df.head(top_n).to_string(index=False))
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compute Fisher Information Matrix for language models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model or HuggingFace model ID')
    parser.add_argument('--calibration_data', type=str, required=True,
                        help='Path to calibration data (JSON, JSONL, or TXT)')
    
    # Optional arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of calibration samples to use')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--output_path', type=str, default='fisher_results.csv',
                        help='Path to save output CSV')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Number of top layers to display')
    
    args = parser.parse_args()
    
    # Print configuration
    print(f"\n{'='*70}")
    print(f"Fisher Information Matrix Calculator")
    print(f"{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Calibration data: {args.calibration_data}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max length: {args.max_length}")
    print(f"Output: {args.output_path}")
    print(f"{'='*70}\n")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,  # Use float32 for stable gradients
        device_map=args.device
    )
    model.eval()
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Disable dropout for consistent Fisher computation
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model loaded")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Load calibration data
    calibration_texts = load_calibration_data(args.calibration_data, args.num_samples)
    
    # Compute Fisher Information
    fisher_dict = compute_fisher_information(
        model=model,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )
    
    # Aggregate by layer
    df = aggregate_fisher_by_layer(fisher_dict)
    
    # Print summary
    print_summary_table(df, top_n=args.top_n)
    
    # Save to CSV
    df.to_csv(args.output_path, index=False)
    print(f"✓ Results saved to: {args.output_path}")
    print(f"\nDone! 🎉")


if __name__ == '__main__':
    main()

