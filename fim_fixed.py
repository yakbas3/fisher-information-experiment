"""
Calculate Fisher Information Matrix (Diagonal) for any PyTorch Model - CORRECTED VERSION

Key fixes:
1. Use float32 instead of float16 to avoid gradient underflow
2. Properly accumulate and normalize Fisher values
3. Better debugging output
4. Handle edge cases
5. FIXED: Compute per-sample gradients instead of batch-averaged gradients
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import argparse
import os
from collections import defaultdict


class TextDataset(Dataset):
    """Simple dataset for text samples"""
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


class FisherCalculator:
    """Calculate diagonal Fisher Information Matrix for any model"""
    
    def __init__(self, model_path, device='cuda:0', use_fp16=False):
        """
        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to use (e.g., 'cuda:0', 'cuda:1', 'cpu')
            use_fp16: Use float16 (can cause numerical issues, use with caution)
        """
        self.device = device
        self.use_fp16 = use_fp16
        print(f"Loading model: {model_path}")
        print(f"Device: {device}")
        print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model - USE FP32 for stable gradients
        dtype = torch.float16 if use_fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        # Set to training mode but disable dropout
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        
        # Print info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✓ Model loaded: {total_params:,} total parameters")
        print(f"✓ Trainable: {trainable_params:,} parameters")
    
    def compute_fisher(self, calibration_texts, batch_size=4, 
                       num_samples=500, max_length=512):
        """
        Compute diagonal Fisher Information Matrix
        
        Args:
            calibration_texts: List of strings for calibration
            batch_size: Batch size
            num_samples: Max number of samples to use
            max_length: Max sequence length
            
        Returns:
            fisher_dict: {parameter_name: fisher_values_tensor}
        """
        print(f"\nComputing Fisher Information...")
        print(f"Samples: {min(len(calibration_texts), num_samples)}")
        
        # Prepare data
        dataset = TextDataset(
            calibration_texts[:num_samples],
            self.tokenizer,
            max_length
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize accumulators - use float32 for numerical stability
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param, dtype=torch.float32)
        
        # Define loss function to get per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none',
            ignore_index=self.tokenizer.pad_token_id
        )
        
        num_batches_processed = 0
        total_samples_processed = 0
        
        # Accumulate Fisher Information
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if total_samples_processed >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            batch_size_actual = input_ids.size(0)
            
            # 1. Forward pass to get logits (DO NOT pass labels)
            with torch.enable_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                
                # 2. Manually compute per-sample loss
                # Shift logits and labels for Causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Get loss for each token
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Reshape to [batch_size, seq_len]
                per_token_loss = per_token_loss.view(batch_size_actual, -1)
                
                # Get number of non-ignored tokens per sample
                non_ignored_tokens = (shift_labels != self.tokenizer.pad_token_id).sum(dim=1)
                # Avoid division by zero
                non_ignored_tokens = torch.clamp(non_ignored_tokens, min=1)
                
                # Calculate the mean loss *per sample*
                per_sample_loss = per_token_loss.sum(dim=1) / non_ignored_tokens
                
                # Debug first batch
                if batch_idx == 0:
                    print(f"\n{'='*70}")
                    print(f"First Batch Debug Info:")
                    print(f"Batch size: {batch_size_actual}")
                    print(f"Per-sample losses: {[f'{x:.4f}' for x in per_sample_loss.detach().cpu().tolist()]}")
                    print(f"{'='*70}\n")
                
                # Check if any loss is invalid
                if torch.isnan(per_sample_loss).any() or torch.isinf(per_sample_loss).any():
                    print(f"Warning: Invalid loss detected in batch {batch_idx}, skipping")
                    continue
                
                # 3. Iterate over each sample to get per-sample gradients
                for j in range(batch_size_actual):
                    self.model.zero_grad()
                    
                    # Backward pass for the j-th sample
                    # Retain graph if not the last sample in the batch
                    is_last_sample = (j == batch_size_actual - 1)
                    per_sample_loss[j].backward(retain_graph=not is_last_sample)
                    
                    # 4. Accumulate squared gradients (Fisher diagonal approximation)
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher_dict[name] += param.grad.detach().to(torch.float32).pow(2)
            
            num_batches_processed += 1
            total_samples_processed += batch_size_actual
        
        # Normalize by number of samples
        print(f"\nNormalizing Fisher values by {total_samples_processed} samples...")
        
        for name in fisher_dict:
            fisher_dict[name] /= total_samples_processed
        
        print(f"✓ Computed Fisher for {total_samples_processed} samples across {num_batches_processed} batches")
        
        # Validate results
        total_fisher = sum(f.sum().item() for f in fisher_dict.values())
        print(f"✓ Total Fisher Information: {total_fisher:.6e}")
        
        if total_fisher == 0 or torch.isnan(torch.tensor(total_fisher)):
            print("\n⚠️  WARNING: Fisher information is zero or NaN!")
            print("   This might indicate:")
            print("   - Numerical precision issues (try use_fp16=False)")
            print("   - Model not properly loaded")
            print("   - Calibration data issues")
        
        return fisher_dict
    
    def aggregate_by_layer(self, fisher_dict):
        """
        Aggregate Fisher values by layer
        
        Returns:
            layer_importance: {layer_id: importance_score}
        """
        layer_importance = defaultdict(float)
        
        for param_name, fisher_values in fisher_dict.items():
            param_importance = fisher_values.sum().item()
            
            # Skip if NaN or inf
            if torch.isnan(torch.tensor(param_importance)) or torch.isinf(torch.tensor(param_importance)):
                print(f"Warning: Skipping {param_name} due to NaN/Inf values")
                continue
            
            layer_id = self._parse_layer_name(param_name)
            layer_importance[layer_id] += param_importance
        
        # Sort and normalize
        total = sum(layer_importance.values())
        
        if total == 0:
            print("Warning: Total importance is zero, cannot normalize")
            return dict(sorted(layer_importance.items()))
        
        normalized = {k: v / total for k, v in layer_importance.items()}
        
        return dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))
    
    def _parse_layer_name(self, param_name):
        """Extract layer identifier from parameter name"""
        if 'embed' in param_name.lower():
            return 'embeddings'
        elif 'lm_head' in param_name or ('output' in param_name and 'layer' not in param_name):
            return 'lm_head'
        elif 'layers.' in param_name or '.layers.' in param_name:
            # Handle different model architectures
            parts = param_name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i+1])
                        return f"layer_{layer_num}"
                    except ValueError:
                        pass
        elif 'norm' in param_name.lower() and 'layer' not in param_name.lower():
            return 'final_norm'
        
        return 'other'
    
    def _parse_component_name(self, param_name):
        """Extract component identifier from parameter name (more granular than layer)"""
        # Examples of components: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, etc.
        
        # Embeddings
        if 'embed' in param_name.lower():
            return 'embeddings'
        
        # LM head
        elif 'lm_head' in param_name or ('output' in param_name and 'layer' not in param_name):
            return 'lm_head'
        
        # Final norm
        elif 'norm' in param_name.lower() and 'layer' not in param_name.lower() and 'layers' not in param_name:
            return 'final_norm'
        
        # Layer-specific components
        elif 'layers.' in param_name or '.layers.' in param_name:
            parts = param_name.split('.')
            
            # Find layer number
            layer_num = None
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i+1])
                        break
                    except ValueError:
                        pass
            
            if layer_num is None:
                return 'other'
            
            # Identify component type
            # Attention components
            if 'q_proj' in param_name:
                return f"layer_{layer_num}.q_proj"
            elif 'k_proj' in param_name:
                return f"layer_{layer_num}.k_proj"
            elif 'v_proj' in param_name:
                return f"layer_{layer_num}.v_proj"
            elif 'o_proj' in param_name or 'out_proj' in param_name:
                return f"layer_{layer_num}.o_proj"
            
            # MLP components
            elif 'gate_proj' in param_name:
                return f"layer_{layer_num}.mlp.gate_proj"
            elif 'up_proj' in param_name:
                return f"layer_{layer_num}.mlp.up_proj"
            elif 'down_proj' in param_name:
                return f"layer_{layer_num}.mlp.down_proj"
            elif 'mlp' in param_name.lower():
                # Generic MLP (for models with different naming)
                if 'fc1' in param_name or 'w1' in param_name:
                    return f"layer_{layer_num}.mlp.fc1"
                elif 'fc2' in param_name or 'w2' in param_name:
                    return f"layer_{layer_num}.mlp.fc2"
                else:
                    return f"layer_{layer_num}.mlp"
            
            # Layer norms
            elif 'input_layernorm' in param_name or 'ln_1' in param_name:
                return f"layer_{layer_num}.input_layernorm"
            elif 'post_attention_layernorm' in param_name or 'ln_2' in param_name:
                return f"layer_{layer_num}.post_attention_layernorm"
            elif 'norm' in param_name.lower():
                return f"layer_{layer_num}.norm"
            
            # Default to just layer if component unclear
            else:
                return f"layer_{layer_num}.other"
        
        return 'other'
    
    def aggregate_by_component(self, fisher_dict, granularity=None):
        """
        Aggregate Fisher values by component (more granular than layer)
        
        Args:
            granularity: If None, aggregate entire components. If int > 0, split each component into N chunks
        
        Returns:
            component_importance: {component_id: {'fisher_sum': value, 'percentage': pct}}
        """
        if granularity is None:
            # Original behavior: aggregate entire components
            component_importance = defaultdict(float)
            
            for param_name, fisher_values in fisher_dict.items():
                param_importance = fisher_values.sum().item()
                
                # Skip if NaN or inf
                if torch.isnan(torch.tensor(param_importance)) or torch.isinf(torch.tensor(param_importance)):
                    print(f"Warning: Skipping {param_name} due to NaN/Inf values")
                    continue
                
                component_id = self._parse_component_name(param_name)
                component_importance[component_id] += param_importance
            
            # Calculate percentages
            total = sum(component_importance.values())
            
            if total == 0:
                print("Warning: Total importance is zero, cannot normalize")
                return dict(sorted(component_importance.items()))
            
            # Return dict with fisher_sum and percentage
            result = {}
            for component_id, fisher_sum in component_importance.items():
                result[component_id] = {
                    'fisher_sum': fisher_sum,
                    'percentage': fisher_sum / total
                }
            
            return dict(sorted(result.items(), key=lambda x: x[1]['fisher_sum'], reverse=True))
        
        else:
            # New behavior: split components into chunks
            print(f"\n🔍 Splitting components into {granularity} chunks each...")
            chunk_importance = {}
            
            for param_name, fisher_values in fisher_dict.items():
                # Skip if NaN or inf
                if torch.isnan(fisher_values).any() or torch.isinf(fisher_values).any():
                    print(f"Warning: Skipping {param_name} due to NaN/Inf values")
                    continue
                
                component_id = self._parse_component_name(param_name)
                
                # Flatten the tensor to 1D for chunking
                fisher_flat = fisher_values.flatten()
                total_elements = fisher_flat.numel()
                
                # Split into N chunks
                chunk_size = max(1, total_elements // granularity)
                
                for chunk_idx in range(granularity):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, total_elements)
                    
                    if start_idx >= total_elements:
                        break
                    
                    # Get chunk and sum Fisher values
                    chunk_fisher = fisher_flat[start_idx:end_idx].sum().item()
                    
                    # Create chunk identifier
                    chunk_id = f"{component_id}[{chunk_idx}]"
                    
                    if chunk_id not in chunk_importance:
                        chunk_importance[chunk_id] = 0.0
                    
                    chunk_importance[chunk_id] += chunk_fisher
            
            # Calculate percentages
            total = sum(chunk_importance.values())
            
            if total == 0:
                print("Warning: Total importance is zero, cannot normalize")
                return dict(sorted(chunk_importance.items()))
            
            # Return dict with fisher_sum and percentage
            result = {}
            for chunk_id, fisher_sum in chunk_importance.items():
                result[chunk_id] = {
                    'fisher_sum': fisher_sum,
                    'percentage': fisher_sum / total
                }
            
            print(f"✓ Created {len(result)} chunks from {len(fisher_dict)} parameters")
            
            return dict(sorted(result.items(), key=lambda x: x[1]['fisher_sum'], reverse=True))
    
    def save_results(self, fisher_dict, layer_importance, component_importance, output_dir, save_fisher_dict=False):
        """Save results to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save layer importance (JSON)
        with open(f'{output_dir}/layer_importance.json', 'w') as f:
            json.dump(layer_importance, f, indent=2)
        
        # Save component importance (JSON) - NEW: more granular than layers
        with open(f'{output_dir}/component_importance.json', 'w') as f:
            json.dump(component_importance, f, indent=2)
        
        # Save full Fisher dict (PyTorch) - OPTIONAL, can be 20+ GB for large models
        if save_fisher_dict:
            print(f"\n💾 Saving full fisher_diagonal.pt (~{sum(p.numel() for p in fisher_dict.values()) * 4 / 1e9:.1f} GB)...")
            torch.save(fisher_dict, f'{output_dir}/fisher_diagonal.pt')
            print("✓ fisher_diagonal.pt saved")
        else:
            print(f"\n⚠️  Skipping fisher_diagonal.pt (would be ~{sum(p.numel() for p in fisher_dict.values()) * 4 / 1e9:.1f} GB)")
            print("   Use --save_fisher_dict to save raw per-parameter Fisher values")
        
        # Save summary statistics
        summary = {
            'total_parameters': len(fisher_dict),
            'total_fisher_information': sum(f.sum().item() for f in fisher_dict.values()),
            'top_10_parameters': [],
            'top_10_components': []
        }
        
        # Get top 10 most important parameters
        param_importance = [(name, fisher.sum().item()) for name, fisher in fisher_dict.items()]
        param_importance.sort(key=lambda x: x[1], reverse=True)
        summary['top_10_parameters'] = param_importance[:10]
        
        # Get top 10 most important components
        component_sorted = sorted(component_importance.items(), key=lambda x: x[1], reverse=True)
        summary['top_10_components'] = component_sorted[:10]
        
        with open(f'{output_dir}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Saved to {output_dir}/")
        print(f"  - layer_importance.json")
        print(f"  - component_importance.json")
        print(f"  - summary.json")
        if save_fisher_dict:
            print(f"  - fisher_diagonal.pt")
    
    def print_results(self, layer_importance):
        """Pretty print results"""
        print("\n" + "="*70)
        print("LAYER IMPORTANCE RANKINGS")
        print("="*70)
        print(f"{'Layer':<20} {'Importance':>15} {'Percentage':>12}")
        print("-"*70)
        
        for layer, importance in layer_importance.items():
            if torch.isnan(torch.tensor(importance)):
                print(f"{layer:<20} {'NaN':>15} {'N/A':>12}")
            else:
                print(f"{layer:<20} {importance:>15.8f} {importance*100:>11.2f}%")
        
        print("="*70)
    
    def print_component_results(self, component_importance, top_n=20, granularity=None):
        """Pretty print component-level results"""
        print("\n" + "="*70)
        if granularity:
            print(f"CHUNK IMPORTANCE RANKINGS (Top {top_n}, Granularity: {granularity})")
        else:
            print(f"COMPONENT IMPORTANCE RANKINGS (Top {top_n})")
        print("="*70)
        print(f"{'Component/Chunk':<50} {'Fisher Sum':>15} {'%':>8}")
        print("-"*70)
        
        # Get top N components/chunks
        sorted_components = sorted(component_importance.items(), 
                                   key=lambda x: x[1]['fisher_sum'] if isinstance(x[1], dict) else x[1], 
                                   reverse=True)
        
        for component, data in sorted_components[:top_n]:
            if isinstance(data, dict):
                fisher_sum = data['fisher_sum']
                percentage = data['percentage']
            else:
                # Backward compatibility (shouldn't happen with new code)
                fisher_sum = data
                percentage = 0
            
            if torch.isnan(torch.tensor(fisher_sum)):
                print(f"{component:<50} {'NaN':>15} {'N/A':>8}")
            else:
                print(f"{component:<50} {fisher_sum:>15.6e} {percentage*100:>7.2f}%")
        
        print("="*70)


def load_calibration_data(file_path):
    """Load calibration data from file (one sample per line)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_sample_calibration_data():
    """Generate sample calibration data for testing"""
    return [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "import torch\nimport torch.nn as nn\n\nclass MyModel(nn.Module):\n    def __init__(self):\n        super().__init__()",
        "for i in range(10):\n    print(f'Iteration {i}')",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:",
        "import numpy as np\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None",
        "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2",
        "try:\n    result = divide(a, b)\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]",
        "import matplotlib.pyplot as plt\nplt.plot(x, y)\nplt.show()",
        "def calculate_mean(numbers):\n    return sum(numbers) / len(numbers)",
        "with open('file.txt', 'r') as f:\n    content = f.read()",
        "lambda x: x ** 2",
        "list_comprehension = [x * 2 for x in range(10)]",
        "def decorator(func):\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)",
        "class Animal:\n    def __init__(self, name):\n        self.name = name",
        "import requests\nresponse = requests.get('https://api.example.com')",
        "def dfs(graph, start, visited=None):\n    if visited is None:\n        visited = set()",
        "result = [x for x in data if x > threshold]",
    ]


def main():
    parser = argparse.ArgumentParser(description='Calculate Fisher Information Matrix')
    parser.add_argument('--model_path', type=str, default='deepseek-ai/deepseek-coder-1.3b-base',
                       help='Path to model or HuggingFace model ID')
    parser.add_argument('--calibration_data', type=str, default=None,
                       help='Path to text file with calibration data')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples to use (default: 20)')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--output_dir', type=str, default='fisher_results',
                       help='Output directory (default: fisher_results)')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use float16 precision (can cause numerical issues)')
    parser.add_argument('--save_fisher_dict', action='store_true',
                       help='Save full per-parameter Fisher dict (can be 20+ GB for large models)')
    parser.add_argument('--granularity', type=int, default=None,
                       help='Split each component into N chunks for fine-grained analysis (e.g., 64). Default: None (no chunking)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Fisher Information Matrix Calculator - CORRECTED VERSION")
    print("="*70)
    
    # Load calibration data
    if args.calibration_data:
        print(f"\nLoading calibration data from: {args.calibration_data}")
        calibration_texts = load_calibration_data(args.calibration_data)
    else:
        print("\nUsing sample calibration data (no file provided)")
        calibration_texts = get_sample_calibration_data()
    
    print(f"✓ Loaded {len(calibration_texts)} samples")
    
    # Initialize calculator
    calculator = FisherCalculator(args.model_path, device=args.device, use_fp16=args.use_fp16)
    
    # Compute Fisher
    fisher_dict = calculator.compute_fisher(
        calibration_texts=calibration_texts,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        max_length=args.max_length
    )
    
    # Aggregate by layer
    layer_importance = calculator.aggregate_by_layer(fisher_dict)
    
    # Aggregate by component (more granular) with optional chunking
    component_importance = calculator.aggregate_by_component(fisher_dict, granularity=args.granularity)
    
    # Print results
    calculator.print_results(layer_importance)
    
    # Print component/chunk results with more items if using granularity
    top_n = 50 if args.granularity else 30
    calculator.print_component_results(component_importance, top_n=top_n, granularity=args.granularity)
    
    # Save results
    calculator.save_results(fisher_dict, layer_importance, component_importance, 
                           args.output_dir, save_fisher_dict=args.save_fisher_dict)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

