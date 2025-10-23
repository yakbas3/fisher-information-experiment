"""
Calculate Task Vector: instruct_model - base_model

This script loads two models (base and instruct) and calculates their difference
to create a task vector that represents the "instruction following" capability.

Usage:
    python calculate_task_vector.py --base_model deepseek-ai/deepseek-coder-1.3b-base \
                                   --instruct_model deepseek-ai/deepseek-coder-1.3b-instruct \
                                   --output_dir task_vectors
"""

import torch
import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


class TaskVectorCalculator:
    """Calculate task vector by subtracting base model from instruct model"""
    
    def __init__(self, base_model_path, instruct_model_path, device='cuda:0'):
        """
        Args:
            base_model_path: Path to base model (HuggingFace ID or local path)
            instruct_model_path: Path to instruct model (HuggingFace ID or local path)
            device: Device to use for computation
        """
        self.device = device
        self.base_model_path = base_model_path
        self.instruct_model_path = instruct_model_path
        
        print("="*70)
        print("Task Vector Calculator")
        print("="*70)
        print(f"Base model: {base_model_path}")
        print(f"Instruct model: {instruct_model_path}")
        print(f"Device: {device}")
        print("="*70)
    
    def load_models(self):
        """Load both models and verify compatibility"""
        print("\nLoading models...")
        
        # Load tokenizers
        print("Loading tokenizers...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self.instruct_tokenizer = AutoTokenizer.from_pretrained(self.instruct_model_path)
        
        # Verify tokenizers are compatible
        if self.base_tokenizer.vocab != self.instruct_tokenizer.vocab:
            print("⚠️  Warning: Tokenizer vocabularies differ between models")
        
        # Load models
        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float32,  # Use float32 for precise calculations
            device_map=self.device,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        print("Loading instruct model...")
        self.instruct_model = AutoModelForCausalLM.from_pretrained(
            self.instruct_model_path,
            torch_dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        # Verify model architectures match
        self._verify_model_compatibility()
        
        print("✓ Models loaded successfully")
    
    def _verify_model_compatibility(self):
        """Verify that both models have the same architecture"""
        base_params = dict(self.base_model.named_parameters())
        instruct_params = dict(self.instruct_model.named_parameters())
        
        # Check parameter names match
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
        
        # Check parameter shapes match
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
    
    def calculate_task_vector(self):
        """Calculate task vector = instruct_model - base_model"""
        print("\nCalculating task vector...")
        
        task_vector = {}
        base_params = dict(self.base_model.named_parameters())
        instruct_params = dict(self.instruct_model.named_parameters())
        
        # Calculate differences for each parameter
        for name in tqdm(base_params.keys(), desc="Computing differences"):
            base_weight = base_params[name]
            instruct_weight = instruct_params[name]
            
            # Calculate difference: instruct - base
            difference = instruct_weight - base_weight
            task_vector[name] = difference.detach().cpu()
        
        print("✓ Task vector calculated")
        return task_vector
    
    def analyze_task_vector(self, task_vector):
        """Analyze the task vector statistics"""
        print("\nAnalyzing task vector...")
        
        analysis = {
            'total_parameters': len(task_vector),
            'parameter_stats': {},
            'layer_stats': {},
            'overall_stats': {}
        }
        
        # Calculate statistics for each parameter
        param_norms = []
        param_means = []
        param_stds = []
        
        for name, diff in task_vector.items():
            diff_tensor = diff.float()
            
            # Basic statistics
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
        
        # Calculate layer statistics
        for layer_id, norms in layer_norms.items():
            analysis['layer_stats'][layer_id] = {
                'total_norm': np.sum(norms),
                'mean_norm': np.mean(norms),
                'max_norm': np.max(norms),
                'parameter_count': len(norms)
            }
        
        return analysis
    
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
    
    def save_task_vector(self, task_vector, analysis, output_dir):
        """Save task vector and analysis to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}/...")
        
        # Save task vector (PyTorch format)
        torch.save(task_vector, f'{output_dir}/task_vector.pt')
        print(f"✓ Task vector saved to {output_dir}/task_vector.pt")
        
        # Save analysis (JSON format)
        with open(f'{output_dir}/task_vector_analysis.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            analysis_serializable = self._make_json_serializable(analysis)
            json.dump(analysis_serializable, f, indent=2)
        print(f"✓ Analysis saved to {output_dir}/task_vector_analysis.json")
        
        # Save model info
        model_info = {
            'base_model': self.base_model_path,
            'instruct_model': self.instruct_model_path,
            'device': self.device,
            'total_parameters': len(task_vector),
            'calculation_timestamp': str(torch.cuda.Event(enable_timing=True).record().elapsed_time(torch.cuda.Event(enable_timing=True).record()))
        }
        
        with open(f'{output_dir}/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✓ Model info saved to {output_dir}/model_info.json")
    
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
    
    def print_analysis(self, analysis):
        """Print analysis results in a readable format"""
        print("\n" + "="*70)
        print("TASK VECTOR ANALYSIS")
        print("="*70)
        
        # Overall statistics
        overall = analysis['overall_stats']
        print(f"\nOverall Statistics:")
        print(f"  Total parameters: {analysis['total_parameters']:,}")
        print(f"  Mean parameter norm: {overall['mean_norm']:.6f}")
        print(f"  Std parameter norm: {overall['std_norm']:.6f}")
        print(f"  Max parameter norm: {overall['max_norm']:.6f}")
        print(f"  Min parameter norm: {overall['min_norm']:.6f}")
        
        # Layer statistics
        print(f"\nLayer-wise Statistics:")
        print(f"{'Layer':<20} {'Total Norm':>15} {'Mean Norm':>15} {'Params':>8}")
        print("-" * 70)
        
        layer_stats = analysis['layer_stats']
        for layer_id in sorted(layer_stats.keys(), key=self._get_sort_key):
            stats = layer_stats[layer_id]
            print(f"{layer_id:<20} {stats['total_norm']:>15.6f} {stats['mean_norm']:>15.6f} {stats['parameter_count']:>8}")
        
        # Top 10 most changed parameters
        print(f"\nTop 10 Most Changed Parameters:")
        print(f"{'Parameter':<40} {'Norm':>15}")
        print("-" * 60)
        
        param_norms = [(name, stats['norm']) for name, stats in analysis['parameter_stats'].items()]
        param_norms.sort(key=lambda x: x[1], reverse=True)
        
        for name, norm in param_norms[:10]:
            print(f"{name:<40} {norm:>15.6f}")
        
        print("="*70)


def get_sort_key(layer_name):
    """Creates a sorting key for layer names."""
    if 'embeddings' in layer_name:
        return -1
    
    import re
    match = re.search(r'layer_(\d+)', layer_name)
    if match:
        return int(match.group(1))
    
    if 'final_norm' in layer_name:
        return 1000
    if 'lm_head' in layer_name:
        return 1001
    return 1002


def main():
    parser = argparse.ArgumentParser(description='Calculate Task Vector (instruct - base)')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Base model path or HuggingFace ID')
    parser.add_argument('--instruct_model', type=str, required=True,
                       help='Instruct model path or HuggingFace ID')
    parser.add_argument('--output_dir', type=str, default='task_vectors',
                       help='Output directory (default: task_vectors)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = TaskVectorCalculator(
        base_model_path=args.base_model,
        instruct_model_path=args.instruct_model,
        device=args.device
    )
    
    # Load models
    calculator.load_models()
    
    # Calculate task vector
    task_vector = calculator.calculate_task_vector()
    
    # Analyze task vector
    analysis = calculator.analyze_task_vector(task_vector)
    
    # Print results
    calculator.print_analysis(analysis)
    
    # Save results
    calculator.save_task_vector(task_vector, analysis, args.output_dir)
    
    print(f"\n✓ Task vector calculation complete!")
    print(f"✓ Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
