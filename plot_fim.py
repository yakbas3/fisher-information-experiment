import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np

def get_sort_key(layer_name):
    """Creates a sorting key for layer names."""
    if 'embeddings' in layer_name:
        return -1
    
    match = re.search(r'layer_(\d+)', layer_name)
    if match:
        return int(match.group(1))
    
    if 'final_norm' in layer_name:
        return 1000
    if 'lm_head' in layer_name:
        return 1001
    return 1002

def plot_importance(json_path, output_image, title):
    """Loads FIM data and saves a Matplotlib bar chart."""
    
    print(f"Loading data from {json_path}...")
    
    try:
        with open(json_path, 'r') as f:
            layer_importance = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(list(layer_importance.items()), columns=['Layer', 'Importance'])

    # Apply and sort by the custom key
    df['sort_key'] = df['Layer'].apply(get_sort_key)
    df_sorted = df.sort_values(by='sort_key').drop(columns='sort_key')
    
    print("Generating plot...")

    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # --- THIS LINE WAS REMOVED ---
    # plt.yscale('log') 
    # --- --------------------- ---
    
    plt.bar(df_sorted['Layer'], df_sorted['Importance'])
    
    plt.title(title, fontsize=16)
    plt.ylabel('Fisher Importance (Linear Scale)')
    plt.xlabel('Model Layer')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)
    
    # Ensure layout is tight so labels are not cut off
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_image, dpi=300)
    
    print(f"✓ Plot saved to {output_image}")


def plot_importance_comparison(json_paths, model_names, output_image, title, colors=None):
    """
    Plot side-by-side comparison of multiple models' layer importance.
    
    Args:
        json_paths: List of paths to layer_importance.json files
        model_names: List of model names for legend
        output_image: Output image path
        title: Chart title
        colors: List of colors for each model (optional)
    """
    print(f"Loading data from {len(json_paths)} files...")
    
    # Load all datasets
    all_data = {}
    for i, json_path in enumerate(json_paths):
        try:
            with open(json_path, 'r') as f:
                layer_importance = json.load(f)
            all_data[model_names[i]] = layer_importance
            print(f"✓ Loaded {model_names[i]}: {len(layer_importance)} layers")
        except FileNotFoundError:
            print(f"Error: Could not find file {json_path}")
            return
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_path}")
            return
    
    # Get all unique layers and sort them
    all_layers = set()
    for data in all_data.values():
        all_layers.update(data.keys())
    
    # Sort layers using the custom key
    sorted_layers = sorted(all_layers, key=get_sort_key)
    
    # Create DataFrame with all models
    df_data = {'Layer': sorted_layers}
    for model_name in model_names:
        df_data[model_name] = [all_data[model_name].get(layer, 0) for layer in sorted_layers]
    
    df = pd.DataFrame(df_data)
    
    print("Generating comparison plot...")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Set up colors
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Calculate bar width and positions
    n_models = len(model_names)
    bar_width = 0.8 / n_models
    x_pos = np.arange(len(sorted_layers))
    
    # Plot bars for each model
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        x_offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x_pos + x_offset, df[model_name], bar_width, 
                     label=model_name, color=color, alpha=0.8)
    
    # Customize the plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Fisher Importance (Linear Scale)', fontsize=12)
    ax.set_xlabel('Model Layer', fontsize=12)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_layers, rotation=90, ha='center')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ensure layout is tight
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    
    print(f"✓ Comparison plot saved to {output_image}")


def plot_importance_single(json_path, output_image, title):
    """Wrapper function for single model plotting (backward compatibility)"""
    plot_importance(json_path, output_image, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot FIM Layer Importance')
    parser.add_argument('--mode', type=str, choices=['single', 'comparison'], default='single',
                        help='Plot mode: single model or comparison of multiple models')
    
    # Single model arguments
    parser.add_argument('--input', type=str,
                        help='Path to the layer_importance.json file (single mode)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the output PNG image (e.g., plot.png)')
    parser.add_argument('--title', type=str, default='FIM Layer Importance',
                        help='Title for the chart')
    
    # Comparison mode arguments
    parser.add_argument('--inputs', type=str, nargs='+',
                        help='List of paths to layer_importance.json files (comparison mode)')
    parser.add_argument('--model_names', type=str, nargs='+',
                        help='List of model names for legend (comparison mode)')
    parser.add_argument('--colors', type=str, nargs='+',
                        help='List of colors for each model (optional)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.input:
            print("Error: --input is required for single mode")
            exit(1)
        plot_importance(args.input, args.output, args.title)
    
    elif args.mode == 'comparison':
        if not args.inputs or not args.model_names:
            print("Error: --inputs and --model_names are required for comparison mode")
            exit(1)
        if len(args.inputs) != len(args.model_names):
            print("Error: Number of inputs must match number of model names")
            exit(1)
        plot_importance_comparison(args.inputs, args.model_names, args.output, args.title, args.colors)