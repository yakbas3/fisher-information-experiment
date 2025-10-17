# Fisher Information Matrix Calculator

Compute diagonal Fisher Information Matrix for PyTorch language models to analyze layer-wise parameter importance.

## Requirements

```bash
pip install torch transformers datasets tqdm
```

## Usage

### Basic Usage

```bash
# With your own data
python fim.py --model_path /path/to/model \
              --calibration_data /path/to/data.txt \
              --num_samples 500

# With built-in samples (testing)
python fim.py --model_path /path/to/model

# Download calibration data from HuggingFace
python load_calibration_dataset.py --output calib.txt --num_samples 500
python fim.py --model_path /path/to/model --calibration_data calib.txt
```

### Key Parameters

**`fim.py`**
```bash
--model_path        Path to model directory or HuggingFace ID
--calibration_data  Text file, one sample per line (optional)
--device           cuda:0, cuda:1, or cpu (default: cuda:0)
--batch_size       Batch size (default: 2)
--num_samples      Number of samples to process (default: 20)
--max_length       Max sequence length in tokens (default: 256)
--output_dir       Output directory (default: fisher_results)
--use_fp16         Enable FP16 (faster, less stable)
```

**`load_calibration_dataset.py`**
```bash
--dataset       codeparrot | the-stack | humaneval | code-contests | code-alpaca
--num_samples   Number of samples (default: 500)
--output        Output file path (default: calibration_data.txt)
--language      Language for the-stack dataset (default: python)
```

### Memory Optimization

| Model Size | VRAM | Batch Size | Max Length |
|-----------|------|------------|------------|
| 1-3B      | 8GB  | 4-8        | 512        |
| 7-13B     | 16GB | 2-4        | 256        |
| 30B+      | 40GB | 1-2        | 128        |

If OOM occurs:
```bash
python fim.py --model_path /path/to/model \
              --batch_size 1 \
              --max_length 128 \
              --num_samples 100
```

## Data Format

Calibration data must be plain text with one sample per line:

```text
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
class Example: pass
for i in range(10): print(i)
```

## Output

Results saved to specified output directory:
- **`layer_importance.json`** - Layer importance rankings
- **`fisher_diagonal.pt`** - Complete Fisher matrix
- **`summary.json`** - Statistics and top-10 parameters

## Examples

```bash
# Standard run
python fim.py --model_path ./my_model --calibration_data data.txt --num_samples 1000

# Limited memory
python fim.py --model_path ./large_model --batch_size 1 --max_length 128

# Multiple GPUs
python fim.py --model_path ./model --device cuda:0 --output_dir results_gpu0
python fim.py --model_path ./model --device cuda:1 --output_dir results_gpu1

# Download dataset
python load_calibration_dataset.py --dataset code-alpaca --num_samples 1000 --output my_calib.txt
```

## Troubleshooting

**CUDA out of memory**: Reduce `--batch_size` or `--max_length`  
**Zero/NaN Fisher values**: Remove `--use_fp16` flag  
**Model loading errors**: Ensure model directory contains config.json and tokenizer files