"""
Comprehensive Model Analysis: Fisher Information + Task Vectors

This script analyzes both base and instruct models by:
1. Calculating Fisher Information Matrix for both models
2. Calculating Task Vector (instruct - base)
3. Comparing layer importance between models
4. Providing comprehensive analysis and visualization

Usage:
    python analyze_models.py --base_model /path/to/base --instruct_model /path/to/instruct \
                            --calibration_data data.txt --num_samples 500
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import argparse
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


class ModelAnalyzer:
    """Comprehensive model analysis combining Fisher Information and Task Vectors"""
    
    def __init__(self, base_model_path, instruct_model_path, device='cuda:0', use_fp16=False):
        self.device = device
        self.use_fp16 = use_fp16
        self.base_model_path = base_model_path
        self.instruct_model_path = instruct_model_path
        
        print("="*80)
        print("COMPREHENSIVE MODEL ANALYSIS")
        print("="*80)
        print(f"Base model: {base_model_path}")
        print(f"Instruct model: {instruct_model_path}")
        print(f"Device: {device}")
        print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
        print("="*80)
    
    def load_models(self):
        """Load both models and verify compatibility"""
        print("\nLoading models...")
        
        # Load tokenizers
        print("Loading tokenizers...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.instruct_tokenizer = AutoTokenizer.from_pretrained(self.instruct_model_path)
        
        if self.base_tokenizer.vocab != self.instruct_tokenizer.vocab:
            print("⚠️  Warning: Tokenizer vocabularies differ between models")
        
        # Load models
        print("Loading base model...")
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        print("Loading instruct model...")
        self.instruct_model = AutoModelForCausalLM.from_pretrained(
            self.instruct_model_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        # Set to training mode but disable dropout
        for model in [self.base_model, self.instruct_model]:
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0
        
        # Verify compatibility
        self._verify_model_compatibility()
        
        print("✓ Models loaded successfully")
    
    def _verify_model_compatibility(self):
        """Verify that both models have the same architecture"""
        base_params = dict(self.base_model.named_parameters())
        instruct_params = dict(self.instruct_model.named_parameters())
        
        base_keys = set(base_params.keys())
        instruct_keys = set(instruct_params.keys())
        
        if base_keys != instruct_keys:
            missing_in_base = instruct_keys - base_keys
            missing_in_instruct = base_keys - instruct_keys
            if missing_in_base:
                print(f"❌ Parameters in instruct but not in base: {missing_in_base}")
            if missing_in_instruct:
                print(f"❌ Parameters in base but not in instruct: {missing_in_instruct}")
            raise ValueError("Model architectures don't match!")
        
        # Check parameter shapes
        shape_mismatches = []
        for name in base_keys:
            if base_params[name].shape != instruct_params[name].shape:
                shape_mismatches.append(f"{name}: base {base_params[name].shape} vs instruct {instruct_params[name].shape}")
        
        if shape_mismatches:
            print("❌ Parameter shape mismatches:")
            for mismatch in shape_mismatches:
                print(f"   {mismatch}")
            raise ValueError("Model parameter shapes don't match!")
        
        print("✓ Model architectures are compatible")
    
    def compute_fisher_information(self, model, model_name, calibration_texts, batch_size=4, 
                                 num_samples=500, max_length=512):
        """Compute Fisher Information Matrix for a model"""
        print(f"\nComputing Fisher Information for {model_name}...")
        print(f"Samples: {min(len(calibration_texts), num_samples)}")
        
        # Prepare data
        dataset = TextDataset(
            calibration_texts[:num_samples],
            self.base_tokenizer,  # Use base tokenizer for consistency
            max_length
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize accumulators
        fisher_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param, dtype=torch.float32)
        
        num_batches_processed = 0
        total_samples_processed = 0
        
        # Accumulate Fisher Information
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {model_name}")):
            if total_samples_processed >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            batch_size_actual = input_ids.size(0)
            
            # Forward pass with gradient tracking
            model.zero_grad()
            
            with torch.enable_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected in batch {batch_idx}, skipping")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Accumulate squared gradients (Fisher diagonal approximation)
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_squared = param.grad.detach().to(torch.float32).pow(2)
                        fisher_dict[name] += grad_squared
            
            num_batches_processed += 1
            total_samples_processed += batch_size_actual
        
        # Normalize by number of samples
        print(f"Normalizing Fisher values by {total_samples_processed} samples...")
        for name in fisher_dict:
            fisher_dict[name] /= total_samples_processed
        
        print(f"✓ Computed Fisher for {model_name}: {total_samples_processed} samples")
        
        # Validate results
        total_fisher = sum(f.sum().item() for f in fisher_dict.values())
        print(f"✓ Total Fisher Information ({model_name}): {total_fisher:.6e}")
        
        if total_fisher == 0 or torch.isnan(torch.tensor(total_fisher)):
            print(f"⚠️  WARNING: Fisher information is zero or NaN for {model_name}!")
        
        return fisher_dict
    
    def calculate_task_vector(self):
        """Calculate task vector = instruct_model - base_model"""
        print("\nCalculating task vector...")
        
        task_vector = {}
        base_params = dict(self.base_model.named_parameters())
        instruct_params = dict(self.instruct_model.named_parameters())
        
        for name in tqdm(base_params.keys(), desc="Computing differences"):
            base_weight = base_params[name]
            instruct_weight = instruct_params[name]
            difference = instruct_weight - base_weight
            task_vector[name] = difference.detach().cpu()
        
        print("✓ Task vector calculated")
        return task_vector
    
    def aggregate_by_layer(self, fisher_dict):
        """Aggregate Fisher values by layer"""
        layer_importance = defaultdict(float)
        
        for param_name, fisher_values in fisher_dict.items():
            param_importance = fisher_values.sum().item()
            
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
    
    def compare_fisher_matrices(self, base_fisher, instruct_fisher):
        """Compare Fisher Information between base and instruct models"""
        print("\nComparing Fisher Information between models...")
        
        comparison = {
            'base_total_fisher': sum(f.sum().item() for f in base_fisher.values()),
            'instruct_total_fisher': sum(f.sum().item() for f in instruct_fisher.values()),
            'fisher_ratio': 0,
            'layer_comparison': {},
            'parameter_comparison': []
        }
        
        # Calculate ratio
        if comparison['base_total_fisher'] > 0:
            comparison['fisher_ratio'] = comparison['instruct_total_fisher'] / comparison['base_total_fisher']
        
        # Compare by layer
        base_layers = self.aggregate_by_layer(base_fisher)
        instruct_layers = self.aggregate_by_layer(instruct_fisher)
        
        all_layers = set(base_layers.keys()) | set(instruct_layers.keys())
        for layer in all_layers:
            base_importance = base_layers.get(layer, 0)
            instruct_importance = instruct_layers.get(layer, 0)
            
            comparison['layer_comparison'][layer] = {
                'base_importance': base_importance,
                'instruct_importance': instruct_importance,
                'difference': instruct_importance - base_importance,
                'ratio': instruct_importance / base_importance if base_importance > 0 else float('inf')
            }
        
        # Compare individual parameters
        for name in base_fisher.keys():
            if name in instruct_fisher:
                base_fisher_val = base_fisher[name].sum().item()
                instruct_fisher_val = instruct_fisher[name].sum().item()
                
                comparison['parameter_comparison'].append({
                    'parameter': name,
                    'base_fisher': base_fisher_val,
                    'instruct_fisher': instruct_fisher_val,
                    'difference': instruct_fisher_val - base_fisher_val,
                    'ratio': instruct_fisher_val / base_fisher_val if base_fisher_val > 0 else float('inf')
                })
        
        # Sort by difference
        comparison['parameter_comparison'].sort(key=lambda x: x['difference'], reverse=True)
        
        return comparison
    
    def analyze_task_vector(self, task_vector):
        """Analyze the task vector statistics"""
        print("\nAnalyzing task vector...")
        
        analysis = {
            'total_parameters': len(task_vector),
            'parameter_stats': {},
            'layer_stats': {},
            'overall_stats': {}
        }
        
        param_norms = []
        param_means = []
        param_stds = []
        
        for name, diff in task_vector.items():
            diff_tensor = diff.float()
            
            norm = torch.norm(diff_tensor).item()
            mean = torch.mean(diff_tensor).item()
            std = torch.std(diff_tensor).item()
            max_val = torch.max(diff_tensor).item()
            min_val = torch.min(diff_tensor).item()
            
            analysis['parameter_stats'][name] = {
                'norm': norm,
                'mean': mean,
                'std': std,
                'max': max_val,
                'min': min_val,
                'shape': list(diff_tensor.shape)
            }
            
            param_norms.append(norm)
            param_means.append(mean)
            param_stds.append(std)
        
        # Overall statistics
        analysis['overall_stats'] = {
            'mean_norm': np.mean(param_norms),
            'std_norm': np.std(param_norms),
            'max_norm': np.max(param_norms),
            'min_norm': np.min(param_norms),
            'mean_parameter_mean': np.mean(param_means),
            'mean_parameter_std': np.mean(param_stds)
        }
        
        # Layer-wise aggregation
        layer_norms = {}
        for name, diff in task_vector.items():
            layer_id = self._parse_layer_name(name)
            norm = torch.norm(diff.float()).item()
            
            if layer_id not in layer_norms:
                layer_norms[layer_id] = []
            layer_norms[layer_id].append(norm)
        
        for layer_id, norms in layer_norms.items():
            analysis['layer_stats'][layer_id] = {
                'total_norm': np.sum(norms),
                'mean_norm': np.mean(norms),
                'max_norm': np.max(norms),
                'parameter_count': len(norms)
            }
        
        return analysis
    
    def save_results(self, base_fisher, instruct_fisher, task_vector, 
                    base_layers, instruct_layers, task_analysis, fisher_comparison, output_dir):
        """Save all results to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}/...")
        
        # Save Fisher Information
        torch.save(base_fisher, f'{output_dir}/base_fisher_diagonal.pt')
        torch.save(instruct_fisher, f'{output_dir}/instruct_fisher_diagonal.pt')
        
        with open(f'{output_dir}/base_layer_importance.json', 'w') as f:
            json.dump(base_layers, f, indent=2)
        
        with open(f'{output_dir}/instruct_layer_importance.json', 'w') as f:
            json.dump(instruct_layers, f, indent=2)
        
        # Save Task Vector
        torch.save(task_vector, f'{output_dir}/task_vector.pt')
        
        with open(f'{output_dir}/task_vector_analysis.json', 'w') as f:
            analysis_serializable = self._make_json_serializable(task_analysis)
            json.dump(analysis_serializable, f, indent=2)
        
        # Save Fisher Comparison
        with open(f'{output_dir}/fisher_comparison.json', 'w') as f:
            comparison_serializable = self._make_json_serializable(fisher_comparison)
            json.dump(comparison_serializable, f, indent=2)
        
        # Save model info
        model_info = {
            'base_model': self.base_model_path,
            'instruct_model': self.instruct_model_path,
            'device': self.device,
            'calculation_timestamp': datetime.datetime.now().isoformat(),
            'total_parameters': len(task_vector)
        }
        
        with open(f'{output_dir}/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✓ All results saved to {output_dir}/")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def print_comprehensive_analysis(self, base_layers, instruct_layers, task_analysis, fisher_comparison):
        """Print comprehensive analysis results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS RESULTS")
        print("="*80)
        
        # Fisher Information Comparison
        print(f"\nFISHER INFORMATION COMPARISON:")
        print(f"  Base model total Fisher: {fisher_comparison['base_total_fisher']:.6e}")
        print(f"  Instruct model total Fisher: {fisher_comparison['instruct_total_fisher']:.6e}")
        print(f"  Fisher ratio (instruct/base): {fisher_comparison['fisher_ratio']:.4f}")
        
        # Task Vector Analysis
        overall = task_analysis['overall_stats']
        print(f"\nTASK VECTOR ANALYSIS:")
        print(f"  Total parameters: {task_analysis['total_parameters']:,}")
        print(f"  Mean parameter norm: {overall['mean_norm']:.6f}")
        print(f"  Max parameter norm: {overall['max_norm']:.6f}")
        print(f"  Min parameter norm: {overall['min_norm']:.6f}")
        
        # Layer importance comparison
        print(f"\nTOP 10 LAYERS BY CHANGE (Task Vector):")
        print(f"{'Layer':<20} {'Base Fisher':>15} {'Instruct Fisher':>15} {'Difference':>15}")
        print("-" * 70)
        
        layer_comp = fisher_comparison['layer_comparison']
        sorted_layers = sorted(layer_comp.items(), key=lambda x: abs(x[1]['difference']), reverse=True)
        
        for layer, stats in sorted_layers[:10]:
            print(f"{layer:<20} {stats['base_importance']:>15.6f} {stats['instruct_importance']:>15.6f} {stats['difference']:>15.6f}")
        
        print("="*80)


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
    ]


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Analysis: Fisher Information + Task Vectors')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Base model path or HuggingFace ID')
    parser.add_argument('--instruct_model', type=str, required=True,
                       help='Instruct model path or HuggingFace ID')
    parser.add_argument('--calibration_data', type=str, default=None,
                       help='Path to text file with calibration data')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to use (default: 100)')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--output_dir', type=str, default='comprehensive_analysis',
                       help='Output directory (default: comprehensive_analysis)')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use float16 precision (can cause numerical issues)')
    
    args = parser.parse_args()
    
    # Load calibration data
    if args.calibration_data:
        print(f"\nLoading calibration data from: {args.calibration_data}")
        calibration_texts = load_calibration_data(args.calibration_data)
    else:
        print("\nUsing sample calibration data (no file provided)")
        calibration_texts = get_sample_calibration_data()
    
    print(f"✓ Loaded {len(calibration_texts)} samples")
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(
        base_model_path=args.base_model,
        instruct_model_path=args.instruct_model,
        device=args.device,
        use_fp16=args.use_fp16
    )
    
    # Load models
    analyzer.load_models()
    
    # Compute Fisher Information for both models
    base_fisher = analyzer.compute_fisher_information(
        analyzer.base_model, "base", calibration_texts,
        batch_size=args.batch_size, num_samples=args.num_samples, max_length=args.max_length
    )
    
    instruct_fisher = analyzer.compute_fisher_information(
        analyzer.instruct_model, "instruct", calibration_texts,
        batch_size=args.batch_size, num_samples=args.num_samples, max_length=args.max_length
    )
    
    # Calculate task vector
    task_vector = analyzer.calculate_task_vector()
    
    # Aggregate by layer
    base_layers = analyzer.aggregate_by_layer(base_fisher)
    instruct_layers = analyzer.aggregate_by_layer(instruct_fisher)
    
    # Analyze task vector
    task_analysis = analyzer.analyze_task_vector(task_vector)
    
    # Compare Fisher matrices
    fisher_comparison = analyzer.compare_fisher_matrices(base_fisher, instruct_fisher)
    
    # Print comprehensive analysis
    analyzer.print_comprehensive_analysis(base_layers, instruct_layers, task_analysis, fisher_comparison)
    
    # Save all results
    analyzer.save_results(
        base_fisher, instruct_fisher, task_vector,
        base_layers, instruct_layers, task_analysis, fisher_comparison, args.output_dir
    )
    
    print(f"\n✓ Comprehensive analysis complete!")
    print(f"✓ Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
