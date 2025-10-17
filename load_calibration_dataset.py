"""
Load calibration data from HuggingFace datasets for Fisher Information calculation

Supports multiple code datasets:
- codeparrot/github-code (clean Python code)
- bigcode/the-stack-dedup (multilingual, massive)
- openai_humaneval (high-quality Python problems)

Usage:
    # Generate calibration file with 500 samples
    python load_calibration_dataset.py --output calibration_data.txt --num_samples 500
    
    # Then use it with your Fisher calculator
    python fim.py --calibration_data calibration_data.txt --num_samples 500
"""

import argparse
from datasets import load_dataset
from tqdm import tqdm
import random


def load_codeparrot(num_samples=500, min_length=100, max_length=2000):
    """
    Load from CodeParrot GitHub Code dataset
    Best for: Clean Python code samples
    """
    print("Loading CodeParrot dataset (Python code)...")
    
    # Load dataset - streaming mode to avoid downloading everything
    dataset = load_dataset(
        "codeparrot/github-code",
        split="train",
        streaming=True,
        languages=["Python"]
    )
    
    samples = []
    for item in tqdm(dataset, total=num_samples, desc="Collecting samples"):
        code = item.get('code', '')
        
        # Filter by length
        if min_length <= len(code) <= max_length:
            samples.append(code)
        
        if len(samples) >= num_samples:
            break
    
    return samples


def load_the_stack(num_samples=500, language="python", min_length=100, max_length=2000):
    """
    Load from The Stack (BigCode)
    Best for: Diverse, multilingual code
    """
    print(f"Loading The Stack dataset ({language})...")
    
    # Load dataset
    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=f"data/{language}",
        split="train",
        streaming=True
    )
    
    samples = []
    for item in tqdm(dataset, total=num_samples, desc="Collecting samples"):
        code = item.get('content', '')
        
        # Filter by length
        if min_length <= len(code) <= max_length:
            samples.append(code)
        
        if len(samples) >= num_samples:
            break
    
    return samples


def load_humaneval(num_samples=164):
    """
    Load OpenAI HumanEval dataset
    Best for: High-quality Python problems (limited to 164 samples)
    """
    print("Loading HumanEval dataset...")
    
    dataset = load_dataset("openai_humaneval", split="test")
    
    samples = []
    for item in dataset:
        # Combine prompt and canonical solution
        code = item['prompt'] + '\n' + item['canonical_solution']
        samples.append(code)
        
        if len(samples) >= num_samples:
            break
    
    return samples


def load_code_contests(num_samples=500, min_length=100, max_length=2000):
    """
    Load from CodeContests dataset
    Best for: Complex algorithmic code
    """
    print("Loading CodeContests dataset...")
    
    dataset = load_dataset(
        "deepmind/code_contests",
        split="train",
        streaming=True
    )
    
    samples = []
    for item in tqdm(dataset, total=num_samples, desc="Collecting samples"):
        # Get Python solutions
        solutions = item.get('solutions', {})
        python_solutions = solutions.get('solution', [])
        
        for code in python_solutions:
            if isinstance(code, str) and min_length <= len(code) <= max_length:
                samples.append(code)
                break
        
        if len(samples) >= num_samples:
            break
    
    return samples


def load_code_alpaca(num_samples=500):
    """
    Load from Code Alpaca dataset
    Best for: Code with instructions/explanations
    """
    print("Loading Code Alpaca dataset...")
    
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    
    # Shuffle for diversity
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    samples = []
    for idx in tqdm(indices[:num_samples], desc="Collecting samples"):
        item = dataset[idx]
        # Combine instruction and output (code)
        code = f"# {item['instruction']}\n{item['output']}"
        samples.append(code)
    
    return samples


def save_samples(samples, output_file):
    """Save samples to file (one per line)"""
    print(f"\nSaving {len(samples)} samples to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            # Clean the sample (remove extra newlines, normalize)
            cleaned = ' '.join(sample.split())  # Remove excess whitespace
            f.write(cleaned + '\n')
    
    print(f"✓ Saved to {output_file}")
    
    # Print statistics
    lengths = [len(s) for s in samples]
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Avg length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Min length: {min(lengths)} chars")
    print(f"  Max length: {max(lengths)} chars")


def main():
    parser = argparse.ArgumentParser(description='Load calibration data from HuggingFace datasets')
    parser.add_argument('--dataset', type=str, default='codeparrot',
                       choices=['codeparrot', 'the-stack', 'humaneval', 'code-contests', 'code-alpaca'],
                       help='Which dataset to use (default: codeparrot)')
    parser.add_argument('--language', type=str, default='python',
                       help='Programming language (for the-stack dataset)')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to collect (default: 500)')
    parser.add_argument('--min_length', type=int, default=100,
                       help='Minimum code length in characters (default: 100)')
    parser.add_argument('--max_length', type=int, default=2000,
                       help='Maximum code length in characters (default: 2000)')
    parser.add_argument('--output', type=str, default='calibration_data.txt',
                       help='Output file path (default: calibration_data.txt)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("="*70)
    print("Calibration Data Loader")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Length range: {args.min_length}-{args.max_length} chars")
    print("="*70)
    
    # Load data based on dataset choice
    try:
        if args.dataset == 'codeparrot':
            samples = load_codeparrot(args.num_samples, args.min_length, args.max_length)
        
        elif args.dataset == 'the-stack':
            samples = load_the_stack(args.num_samples, args.language, args.min_length, args.max_length)
        
        elif args.dataset == 'humaneval':
            samples = load_humaneval(min(args.num_samples, 164))
        
        elif args.dataset == 'code-contests':
            samples = load_code_contests(args.num_samples, args.min_length, args.max_length)
        
        elif args.dataset == 'code-alpaca':
            samples = load_code_alpaca(args.num_samples)
        
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        if len(samples) == 0:
            print("\n⚠️  No samples collected! Try different parameters.")
            return
        
        # Save to file
        save_samples(samples, args.output)
        
        print("\n✓ Done! Now run:")
        print(f"   python fim.py --calibration_data {args.output} --num_samples {len(samples)}")
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Install datasets: pip install datasets")
        print("  2. Some datasets require authentication:")
        print("     - Login: huggingface-cli login")
        print("     - Accept terms on HuggingFace website")
        print("  3. Try a different dataset with --dataset flag")


if __name__ == "__main__":
    main()