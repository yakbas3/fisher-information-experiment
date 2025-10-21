import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse

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
    
    # Use a log scale for the y-axis to see small values
    # Remove this line if you prefer a linear scale
    plt.yscale('log') 
    
    plt.bar(df_sorted['Layer'], df_sorted['Importance'])
    
    plt.title(title, fontsize=16)
    plt.ylabel('Fisher Importance (Log Scale)')
    plt.xlabel('Model Layer')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)
    
    # Ensure layout is tight so labels are not cut off
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_image, dpi=300)
    
    print(f"✓ Plot saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot FIM Layer Importance')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the layer_importance.json file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the output PNG image (e.g., plot.png)')
    parser.add_argument('--title', type=str, default='FIM Layer Importance',
                        help='Title for the chart')
    
    args = parser.parse_args()
    
    plot_importance(args.input, args.output, args.title)