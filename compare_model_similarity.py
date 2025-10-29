"""
Memory-Efficient Model Similarity Analyzer for Three LLMs

This script compares a base model and two fine-tunes by:
1. Loading one tensor component at a time (memory-efficient)
2. Computing column-wise cosine similarity
3. Aggregating results into mean values
4. Generating heatmaps and line plots

Usage:
    python compare_model_similarity.py
"""

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import os
import argparse


# ============================================================================
# Helper Functions
# ============================================================================

def get_safetensors_index(repo_id):
    """
    Fetch the model's safetensors index (table of contents).
    
    Args:
        repo_id: HuggingFace repository ID
        
    Returns:
        weight_map: Dictionary mapping tensor keys to shard filenames
    """
    print(f"Fetching safetensors index for {repo_id}...")
    
    try:
        # Download the index file
        index_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            cache_dir=None
        )
        
        # Load and parse the index
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get('weight_map', {})
        print(f"✓ Found {len(weight_map)} tensors in {repo_id}")
        
        return weight_map
    
    except Exception as e:
        # If no index file exists, the model might be in a single safetensors file
        print(f"No index file found for {repo_id}, trying single file...")
        try:
            # Try to download the single safetensors file
            single_file = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                cache_dir=None
            )
            
            # Load the file to get all keys
            tensors = load_file(single_file)
            weight_map = {key: "model.safetensors" for key in tensors.keys()}
            print(f"✓ Found {len(weight_map)} tensors in single file for {repo_id}")
            
            return weight_map
        
        except Exception as e2:
            print(f"❌ Error loading model index: {e2}")
            raise


def load_specific_tensor(repo_id, tensor_key, weight_map):
    """
    Load a single tensor into memory efficiently.
    
    Args:
        repo_id: HuggingFace repository ID
        tensor_key: Key of the tensor to load
        weight_map: Dictionary mapping tensor keys to shard filenames
        
    Returns:
        tensor: The loaded tensor on the specified device in float32
    """
    # Find the shard file containing this tensor
    shard_filename = weight_map.get(tensor_key)
    
    if shard_filename is None:
        raise KeyError(f"Tensor {tensor_key} not found in weight_map")
    
    # Download the shard file (cached automatically)
    shard_path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_filename,
        cache_dir=None
    )
    
    # Load only the specific tensor from the shard
    tensors = load_file(shard_path)
    tensor = tensors[tensor_key]
    
    # Move to device and convert to float32
    tensor = tensor.to(device=DEVICE, dtype=torch.float32)
    
    return tensor


def calculate_mean_column_similarity(tensor_a, tensor_b):
    """
    Calculate column-wise cosine similarity and return the mean.
    
    Args:
        tensor_a: First tensor (2D matrix)
        tensor_b: Second tensor (2D matrix)
        
    Returns:
        mean_similarity: Scalar mean of column-wise similarities
    """
    # Ensure tensors are 2D
    if tensor_a.dim() != 2 or tensor_b.dim() != 2:
        raise ValueError(f"Tensors must be 2D. Got shapes: {tensor_a.shape}, {tensor_b.shape}")
    
    # Ensure same shape
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Tensors must have same shape. Got: {tensor_a.shape}, {tensor_b.shape}")
    
    # Compute column-wise cosine similarity (dim=0 means along rows, comparing columns)
    column_similarities = F.cosine_similarity(tensor_a, tensor_b, dim=0)
    
    # Return the mean as a scalar
    mean_similarity = column_similarities.mean().item()
    
    return mean_similarity


def parse_tensor_key(tensor_key):
    """
    Parse tensor key to extract layer number and component type.
    
    Args:
        tensor_key: Full tensor key (e.g., "model.layers.5.self_attn.q_proj.weight")
        
    Returns:
        layer_number: Layer number (int or None)
        component_type: Component type string
    """
    # Try to extract layer number
    layer_match = re.search(r'layers?[.\[](\d+)', tensor_key)
    layer_number = int(layer_match.group(1)) if layer_match else None
    
    # Extract component type based on suffix
    component_type = "unknown"
    for suffix in TENSOR_SUFFIXES_TO_COMPARE:
        if tensor_key.endswith(suffix):
            # Get the component name (e.g., "q_proj", "gate_proj")
            component_parts = suffix.split('.')
            if len(component_parts) >= 2:
                # Format: "self_attn.q_proj" or "mlp.gate_proj"
                component_type = f"{component_parts[-3]}.{component_parts[-2]}" if len(component_parts) >= 3 else component_parts[-2]
            break
    
    return layer_number, component_type


# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(description='Memory-Efficient Model Similarity Analyzer')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Base model repository ID (e.g., deepseek-ai/deepseek-coder-1.3b-base)')
    parser.add_argument('--finetune1', type=str, required=True,
                       help='First fine-tune model repository ID')
    parser.add_argument('--finetune2', type=str, required=True,
                       help='Second fine-tune model repository ID')
    parser.add_argument('--output_dir', type=str, default='similarity_analysis',
                       help='Output directory (default: similarity_analysis)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use: cuda, cpu, cuda:0, etc. (default: auto-detect)')
    parser.add_argument('--components', type=str, nargs='+', 
                       default=["self_attn.q_proj.weight", "self_attn.k_proj.weight", 
                               "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                               "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"],
                       help='Tensor suffixes to compare (default: standard transformer components)')
    
    args = parser.parse_args()
    
    # Configuration from arguments
    BASE_MODEL_REPO_ID = args.base_model
    FINETUNE1_REPO_ID = args.finetune1
    FINETUNE2_REPO_ID = args.finetune2
    OUTPUT_DIR = args.output_dir
    DEVICE = args.device
    TENSOR_SUFFIXES_TO_COMPARE = args.components
    
    print("="*80)
    print("Memory-Efficient Model Similarity Analyzer")
    print("="*80)
    print(f"Base Model: {BASE_MODEL_REPO_ID}")
    print(f"Fine-tune 1: {FINETUNE1_REPO_ID}")
    print(f"Fine-tune 2: {FINETUNE2_REPO_ID}")
    print(f"Device: {DEVICE}")
    print(f"Components: {len(TENSOR_SUFFIXES_TO_COMPARE)}")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Get weight maps for all three models
    print("\n[1/5] Fetching model indices...")
    base_weight_map = get_safetensors_index(BASE_MODEL_REPO_ID)
    ft1_weight_map = get_safetensors_index(FINETUNE1_REPO_ID)
    ft2_weight_map = get_safetensors_index(FINETUNE2_REPO_ID)
    
    # Step 2: Initialize results storage
    results_list = []
    
    # Filter base model keys to only those we care about
    target_keys = [
        key for key in base_weight_map.keys()
        if any(key.endswith(suffix) for suffix in TENSOR_SUFFIXES_TO_COMPARE)
    ]
    
    print(f"\n[2/5] Found {len(target_keys)} target tensors to compare")
    
    # Step 3: Processing loop
    print("\n[3/5] Processing tensors (memory-efficient loop)...")
    
    successful_comparisons = 0
    failed_comparisons = 0
    
    for tensor_key in tqdm(target_keys, desc="Comparing tensors"):
        try:
            # Load tensors from all three models
            tensor_base = load_specific_tensor(BASE_MODEL_REPO_ID, tensor_key, base_weight_map)
            
            # Check if tensor exists in both fine-tunes
            if tensor_key not in ft1_weight_map or tensor_key not in ft2_weight_map:
                # Skip if tensor doesn't exist in all three models
                del tensor_base
                torch.cuda.empty_cache()
                continue
            
            tensor_ft1 = load_specific_tensor(FINETUNE1_REPO_ID, tensor_key, ft1_weight_map)
            tensor_ft2 = load_specific_tensor(FINETUNE2_REPO_ID, tensor_key, ft2_weight_map)
            
            # Calculate column-wise similarities
            mean_base_ft1 = calculate_mean_column_similarity(tensor_base, tensor_ft1)
            mean_base_ft2 = calculate_mean_column_similarity(tensor_base, tensor_ft2)
            mean_ft1_ft2 = calculate_mean_column_similarity(tensor_ft1, tensor_ft2)
            
            # Parse tensor key
            layer_number, component_type = parse_tensor_key(tensor_key)
            
            # Store results
            results_list.append({
                'tensor_key': tensor_key,
                'layer': layer_number,
                'component_type': component_type,
                'base_vs_ft1': mean_base_ft1,
                'base_vs_ft2': mean_base_ft2,
                'ft1_vs_ft2': mean_ft1_ft2,
                'tensor_shape': str(tensor_base.shape)
            })
            
            successful_comparisons += 1
            
            # Critical: Free memory before next iteration
            del tensor_base, tensor_ft1, tensor_ft2
            torch.cuda.empty_cache()
            
        except Exception as e:
            failed_comparisons += 1
            tqdm.write(f"⚠️  Skipping {tensor_key}: {str(e)[:100]}")
            # Ensure memory is freed even on error
            if 'tensor_base' in locals():
                del tensor_base
            if 'tensor_ft1' in locals():
                del tensor_ft1
            if 'tensor_ft2' in locals():
                del tensor_ft2
            torch.cuda.empty_cache()
            continue
    
    print(f"\n✓ Successful comparisons: {successful_comparisons}")
    print(f"✗ Failed comparisons: {failed_comparisons}")
    
    # Step 4: Convert to DataFrame
    print("\n[4/5] Converting results to DataFrame...")
    df = pd.DataFrame(results_list)
    
    # Filter out rows with None layer numbers (embedding layers, etc.)
    df_layers = df[df['layer'].notna()].copy()
    df_layers['layer'] = df_layers['layer'].astype(int)
    
    # Save raw results
    df.to_csv(f"{OUTPUT_DIR}/similarity_results.csv", index=False)
    print(f"✓ Saved raw results to {OUTPUT_DIR}/similarity_results.csv")
    print(f"✓ Total components compared: {len(df)}")
    print(f"✓ Layer components: {len(df_layers)}")
    
    # Step 5: Generate visualizations
    print("\n[5/5] Generating visualizations...")
    
    if len(df_layers) == 0:
        print("⚠️  No layer data to visualize!")
        return
    
    # ========================================================================
    # Plot 1: Heatmap Grid
    # ========================================================================
    print("\nGenerating heatmap grid...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Model Similarity Heatmaps (Column-wise Cosine Similarity)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Define comparison columns and titles
    comparisons = [
        ('base_vs_ft1', 'Base vs Fine-tune 1'),
        ('base_vs_ft2', 'Base vs Fine-tune 2'),
        ('ft1_vs_ft2', 'Fine-tune 1 vs Fine-tune 2')
    ]
    
    # Shared color scale for comparability
    vmin, vmax = 0.8, 1.0
    
    for idx, (col_name, title) in enumerate(comparisons):
        ax = axes[idx]
        
        # Create pivot table
        pivot = df_layers.pivot_table(
            index='component_type',
            columns='layer',
            values=col_name,
            aggfunc='mean'
        )
        
        # Plot heatmap
        sns.heatmap(
            pivot,
            ax=ax,
            cmap='RdYlGn',
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'Mean Column Similarity'},
            annot=False,
            fmt='.3f',
            linewidths=0.5
        )
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Layer Number', fontsize=10)
        ax.set_ylabel('Component Type', fontsize=10)
    
    plt.tight_layout()
    heatmap_path = f"{OUTPUT_DIR}/similarity_heatmaps.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved heatmap to {heatmap_path}")
    
    # ========================================================================
    # Plot 2: Faceted Line Plot
    # ========================================================================
    print("\nGenerating faceted line plot...")
    
    # Melt the DataFrame to long format
    df_melted = df_layers.melt(
        id_vars=['layer', 'component_type', 'tensor_key'],
        value_vars=['base_vs_ft1', 'base_vs_ft2', 'ft1_vs_ft2'],
        var_name='comparison',
        value_name='similarity'
    )
    
    # Rename comparison labels for better readability
    comparison_labels = {
        'base_vs_ft1': 'Base vs FT1',
        'base_vs_ft2': 'Base vs FT2',
        'ft1_vs_ft2': 'FT1 vs FT2'
    }
    df_melted['comparison'] = df_melted['comparison'].map(comparison_labels)
    
    # Create faceted line plot
    g = sns.relplot(
        data=df_melted,
        x='layer',
        y='similarity',
        hue='comparison',
        col='component_type',
        kind='line',
        col_wrap=3,
        height=4,
        aspect=1.2,
        markers=True,
        dashes=False,
        palette='Set2',
        linewidth=2,
        markersize=6,
        alpha=0.8
    )
    
    # Set consistent y-axis limits
    g.set(ylim=(0.8, 1.0))
    
    # Customize titles and labels
    g.set_axis_labels("Layer Number", "Mean Column Similarity", fontsize=11)
    g.set_titles("{col_name}", fontsize=12, fontweight='bold')
    
    # Add overall title
    g.fig.suptitle('Layer-wise Similarity Comparison Across Components', 
                   fontsize=14, fontweight='bold', y=1.00)
    
    # Adjust legend
    g._legend.set_title('Comparison')
    g._legend.set_bbox_to_anchor((1.05, 0.5))
    
    plt.tight_layout()
    lineplot_path = f"{OUTPUT_DIR}/similarity_lineplot.png"
    g.savefig(lineplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved line plot to {lineplot_path}")
    
    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for col_name, title in comparisons:
        mean_val = df_layers[col_name].mean()
        std_val = df_layers[col_name].std()
        min_val = df_layers[col_name].min()
        max_val = df_layers[col_name].max()
        
        print(f"\n{title}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  Min:  {min_val:.4f}")
        print(f"  Max:  {max_val:.4f}")
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"✓ Results saved to: {OUTPUT_DIR}/")
    print("="*80)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()

