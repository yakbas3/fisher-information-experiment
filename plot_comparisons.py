#!/usr/bin/env python3
"""
Helper script to generate comparison plots for Fisher Information experiments.

This script creates side-by-side comparison plots for:
1. GSM8K benchmark: Math model vs Coder model
2. HumanEval benchmark: Math model vs Coder model

Usage:
    python plot_comparisons.py --results_dir ./results
"""

import argparse
import os
import glob
from plot_fim import plot_importance_comparison

def find_result_files(results_dir, benchmark, model_type):
    """Find layer_importance.json files for a specific benchmark and model type."""
    # Try both patterns: model_benchmark and benchmark_model
    patterns = [
        os.path.join(results_dir, f"*{model_type}*{benchmark}*", "layer_importance.json"),
        os.path.join(results_dir, f"*{benchmark}*{model_type}*", "layer_importance.json")
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots for FIM experiments')
    parser.add_argument('--results_dir', type=str, default='.',
                       help='Base directory containing result folders')
    parser.add_argument('--output_dir', type=str, default='comparison_plots',
                       help='Directory to save comparison plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define experiments
    experiments = [
        {
            'name': 'GSM8K',
            'benchmark': 'gsm8k',
            'title': 'Layer Importance Comparison - GSM8K (Math Reasoning)',
            'filename': 'gsm8k_comparison.png'
        },
        {
            'name': 'HumanEval',
            'benchmark': 'humaneval',
            'title': 'Layer Importance Comparison - HumanEval (Code Generation)',
            'filename': 'humaneval_comparison.png'
        }
    ]
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Generating {exp['name']} comparison plot...")
        print(f"{'='*60}")
        
        # Find result files
        math_file = find_result_files(args.results_dir, exp['benchmark'], 'math')
        coder_file = find_result_files(args.results_dir, exp['benchmark'], 'coder')
        
        if not math_file:
            print(f"❌ Could not find Math model results for {exp['name']}")
            print(f"   Looking for patterns: *math*{exp['benchmark']}* or *{exp['benchmark']}*math*")
            continue
            
        if not coder_file:
            print(f"❌ Could not find Coder model results for {exp['name']}")
            print(f"   Looking for patterns: *coder*{exp['benchmark']}* or *{exp['benchmark']}*coder*")
            continue
        
        print(f"✓ Found Math model results: {math_file}")
        print(f"✓ Found Coder model results: {coder_file}")
        
        # Generate comparison plot
        output_path = os.path.join(args.output_dir, exp['filename'])
        
        try:
            plot_importance_comparison(
                json_paths=[math_file, coder_file],
                model_names=['DeepSeek Math 7B', 'DeepSeek Coder 7B'],
                output_image=output_path,
                title=exp['title'],
                colors=['#2E8B57', '#4169E1']  # Sea Green and Royal Blue
            )
            print(f"✓ {exp['name']} comparison plot saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error generating {exp['name']} plot: {e}")
    
    print(f"\n{'='*60}")
    print("Comparison plot generation complete!")
    print(f"Check the '{args.output_dir}' directory for results.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
