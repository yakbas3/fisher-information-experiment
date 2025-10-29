# Fisher Information Matrix Calculator & Model Similarity Analyzer

A comprehensive toolkit for analyzing PyTorch language models:
- **Fisher Information Matrix (FIM)**: Identify which parameters are most important
- **Model Similarity Analyzer**: Compare three models efficiently to see what changed during fine-tuning

## Tools in This Repository

### 1. Fisher Information Matrix Calculator (`fim_fixed.py`)
Compute diagonal Fisher Information Matrix to analyze layer-wise parameter importance.

### 2. Model Similarity Analyzer (`compare_model_similarity.py`) ⭐ NEW
Memory-efficient column-wise cosine similarity analysis comparing three models.

### 3. Comprehensive Analysis (`analyze_models.py`)
Combined Fisher Information + Task Vector analysis for base vs instruct models.

## Requirements

### For Fisher Information Matrix:
```bash
pip install torch transformers datasets tqdm matplotlib pandas seaborn
```

### For Model Similarity Analyzer:
```bash
pip install -r requirements_similarity.txt
```
or
```bash
pip install torch huggingface_hub safetensors pandas seaborn matplotlib tqdm
```

## Quick Start

### Model Similarity Analyzer (Fastest Start! ⚡)

```bash
# 1. Install dependencies
pip install -r requirements_similarity.txt

# 2. Run analysis (edit model IDs in the script first)
python compare_model_similarity.py

# 3. Check results in similarity_analysis/ folder
```

📖 **See [QUICKSTART_similarity.md](QUICKSTART_similarity.md) for detailed guide**

### Fisher Information Matrix

```bash
# With your own data
python fim_fixed.py --model_path /path/to/model \
                    --calibration_data /path/to/data.txt \
                    --num_samples 500

# With built-in samples (testing)
python fim_fixed.py --model_path /path/to/model

# Download calibration data from HuggingFace
python load_calibration_dataset.py --output calib.txt --num_samples 500
python fim_fixed.py --model_path /path/to/model --calibration_data calib.txt
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

### Fisher Information Matrix
**CUDA out of memory**: Reduce `--batch_size` or `--max_length`  
**Zero/NaN Fisher values**: Remove `--use_fp16` flag  
**Model loading errors**: Ensure model directory contains config.json and tokenizer files

### Model Similarity Analyzer
**Out of memory**: Set `DEVICE = "cpu"` in the script  
**Model not found**: Check repository ID on Hugging Face Hub  
**Tensor shape mismatch**: Models must have same architecture

## Advanced Usage

### Combining Fisher Information + Similarity Analysis

```bash
# Step 1: Compute Fisher Information for both models
python fim_fixed.py --model_path model/base --output_dir fim_base --num_samples 200
python fim_fixed.py --model_path model/instruct --output_dir fim_instruct --num_samples 200

# Step 2: Compute similarity between models
python compare_model_similarity.py  # Edit model IDs first

# Step 3: Analyze results together
# - FIM shows which parameters are important
# - Similarity shows which parameters changed
# - Important parameters that changed significantly = key insights!
```

### Complete Analysis Workflow

```bash
# 1. Comprehensive analysis (Fisher + Task Vectors)
python analyze_models.py --base_model model/base \
                        --instruct_model model/instruct \
                        --calibration_data calib.txt \
                        --num_samples 200

# 2. Similarity analysis (compare 3 models)
python compare_model_similarity.py

# 3. Generate plots
python plot_comparisons.py --results_dir ./
```

## Documentation

- **[QUICKSTART_similarity.md](QUICKSTART_similarity.md)** - Get started with similarity analysis in 5 minutes
- **[README_similarity_analyzer.md](README_similarity_analyzer.md)** - Detailed documentation for similarity analyzer
- **[similarity_config_example.py](similarity_config_example.py)** - Configuration examples and tips

## Project Structure

```
fisher-information-experiment/
├── fim_fixed.py                    # FIM calculator (corrected, per-sample gradients)
├── fim.py                          # FIM calculator (original version)
├── compare_model_similarity.py     # NEW: Memory-efficient similarity analyzer
├── analyze_models.py               # Comprehensive FIM + Task Vector analysis
├── calculate_task_vector.py        # Standalone task vector calculator
├── load_calibration_dataset.py     # Download calibration data
├── plot_fim.py                     # Visualization tools
├── plot_comparisons.py             # Generate comparison plots
├── README.md                       # This file
├── README_similarity_analyzer.md   # Detailed similarity analyzer docs
├── QUICKSTART_similarity.md        # Quick start guide
├── similarity_config_example.py    # Example configurations
├── requirements.txt                # FIM requirements
└── requirements_similarity.txt     # Similarity analyzer requirements
```

## Key Features

### Fisher Information Matrix Calculator
- ✅ Accurate per-sample gradient computation
- ✅ Component-level and layer-level aggregation
- ✅ Optional fine-grained chunking analysis
- ✅ Multiple dataset loaders (CodeParrot, The Stack, HumanEval, GSM8K)
- ✅ Task vector support

### Model Similarity Analyzer ⭐
- ✅ **Extremely memory-efficient** (loads one tensor at a time)
- ✅ Column-wise cosine similarity
- ✅ Compare 3 models simultaneously
- ✅ Automatic aggregation and visualization
- ✅ Heatmap grid + faceted line plots
- ✅ Works with any Hugging Face model

## Use Cases

1. **Understand Fine-tuning Impact**: See which layers/components changed most
2. **Compare Training Strategies**: Evaluate different fine-tuning approaches
3. **Model Merging**: Check compatibility before merging models
4. **Transfer Learning**: Identify which layers to freeze vs fine-tune
5. **Parameter Importance**: Find critical parameters using Fisher Information
6. **Optimization**: Focus compression/pruning on less important parameters

## Citation

If you use this code in your research, please cite appropriately and consider sharing your findings!