# Digital Archives - LLM-Based Entity Recognition

This project implements a comprehensive framework for extracting named entities from historical digital archive materials using Large Language Models (LLMs). It was developed as part of the ARA (Archives and Records Association) research grant application "LUCAS" (LLM Understanding and Classification of Archive Sources).

## Project Overview

This toolkit enables researchers to perform custom Named Entity Recognition (NER) on historical documents, particularly focusing on slavery-related archival materials. The system uses local LLMs via LM Studio to identify and classify entities such as:

- **PERSON**: Individual names
- **PLACE**: Geographic locations
- **DATE**: Temporal references
- **ORG**: Organisations and institutions
- **SHIP**: Vessel names
- **GROUP**: Social or family groups
- **INDICATION_OF_SLAVERY**: Content indicating slavery or slave trade
- **RELATIONSHIPS**: Social, familial, or business relationships

## Key Features

- **Local LLM Integration**: Uses LM Studio for privacy-preserving, offline processing
- **Multiple Model Support**: Compatible with various models (Llama, Gemma, etc.)
- **Batch Processing**: Handles large documents by intelligently chunking text
- **Entity Deduplication**: Removes duplicate entities across document chunks
- **Comparative Analysis**: Tools to compare model performance
- **Visualization**: Interactive highlighting of extracted entities
- **Excel Export**: Converts JSON results to Excel format for analysis

## Architecture

```
lm_studio/
├── run_ner.py              # Main NER processing script
├── run_ner.sh              # Bash script for easy model execution
├── compare_results.py      # Model comparison and evaluation
├── json_to_excel.py        # Convert JSON outputs to Excel
├── visualize_ner.py        # Visualization tools
└── outputs/                # Experimental results by model
    ├── experiment_1/
    ├── experiment_2/
    └── experiment_3/
```

## Prerequisites

### System Requirements

- Python 3.8+
- LM Studio installed and running locally
- 8GB+ RAM (16GB+ recommended for larger models)

### LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download your preferred models (e.g., Llama 3.3 70B, Gemma 2 27B)
3. Start the local server (default: `http://127.0.0.1:1234/v1`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd clean-digital-archives
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Entity Extraction

1. **Prepare your text files:**
   - Place `.txt` files in an input directory
   - Ensure files are UTF-8 encoded

2. **Run entity extraction:**
   ```bash
   # Using the shell script (recommended)
   ./lm_studio/run_ner.sh "llama-3.3-70b-instruct" input_directory output_directory
   
   # Or directly with Python
   python lm_studio/run_ner.py "llama-3.3-70b-instruct" input_directory output_directory
   ```

3. **Results:** JSON files with extracted entities will be saved to the output directory.

### Advanced Usage

#### Processing Large Documents

The system automatically handles large documents by:
- Estimating token count (1 token ≈ 4 characters)
- Splitting into overlapping chunks (max 12,000 tokens per chunk)
- Deduplicating entities across chunks

#### Custom Entity Categories

Modify the `NER_TOOL` definition in `run_ner.py` to add or change entity categories:

```python
"enum": [
    "DATE", 
    "PERSON", 
    "ORG", 
    "PLACE", 
    "SHIP", 
    "GROUP", 
    "INDICATION_OF_SLAVERY", 
    "RELATIONSHIPS",
    "YOUR_CUSTOM_CATEGORY"  # Add your categories here
]
```

#### Batch Processing Multiple Models

```bash
# Process with multiple models
for model in "llama-3.3-70b-instruct" "gemma-2-27b-it" "llama-4-scout-17b"
do
    ./lm_studio/run_ner.sh "$model" input_dir "outputs/experiment_1/$model"
done
```

## Analysis and Visualization

### Convert to Excel

```bash
python lm_studio/json_to_excel.py lm_studio/outputs/experiment_1
```

### Compare Model Performance

```bash
python lm_studio/compare_results.py --ground-truth datasets/labelled_data/ground_truth.xlsx --show-examples
```

### Visualize Entities

```python
from lm_studio.visualize_ner import highlight_entities
import json

# Load your results
with open('output_entities.json', 'r') as f:
    results = json.load(f)

# Create highlighted HTML
highlighted_text = highlight_entities(original_text, results['entities'])
```

## Model Recommendations

Based on experimental results:

- **Llama 3.3 70B**: Best overall performance, high accuracy
- **Gemma 2 27B**: Good balance of speed and accuracy
- **Llama 4 Scout 17B**: Faster processing, suitable for large datasets

## Output Format

Entity extraction results are saved as JSON:

```json
{
  "entities": [
    {
      "text": "William Roscoe",
      "label": "PERSON"
    },
    {
      "text": "Liverpool",
      "label": "PLACE"
    },
    {
      "text": "1753-1831",
      "label": "DATE"
    }
  ]
}
```

## Configuration

Key configuration options in `run_ner.py`:

```python
MAX_TOKENS_PER_CHUNK = 12000    # Maximum tokens per processing chunk
OVERLAP_TOKENS = 500            # Overlap between chunks
```

## Troubleshooting

### Common Issues

1. **LM Studio Connection Error:**
   - Ensure LM Studio is running and the server is started
   - Check the base URL: `http://127.0.0.1:1234/v1`

2. **Out of Memory:**
   - Reduce `MAX_TOKENS_PER_CHUNK`
   - Use a smaller model
   - Process files individually

3. **JSON Parsing Errors:**
   - The system includes fallback parsers for malformed JSON
   - Check debug files for raw model outputs

### Performance Optimization

- **Memory:** Use 16GB+ RAM for 70B models
- **Speed:** Smaller models (7B-27B) for faster processing
- **Accuracy:** Larger models (70B+) for better entity recognition

## Research Applications

This toolkit has been specifically designed for:

- Historical document analysis
- Slavery and colonial archive research
- Genealogical research
- Social network analysis of historical figures
- Maritime history research

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is released under the MIT License. See `LICENSE` file for details.

## Citation

If you use this toolkit in your research, please cite:

```
Digital Archives: LLM-Based Entity Recognition for Historical Documents
ARA Research Grant Application LUCAS (LLM Understanding and Classification of Archive Sources)
2024
```

## Support and Contact

For questions, issues, or collaboration opportunities, please:

1. Open an issue on GitHub
2. Contact the research team via the ARA grant application

## Acknowledgments

This project was developed as part of the Archives and Records Association (ARA) research grant program, focusing on the application of Large Language Models to historical archive materials, particularly those related to slavery and colonial history. 