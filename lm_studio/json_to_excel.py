import json
import os
import pandas as pd
from collections import defaultdict
import argparse

# Argument parsing for experiment folder
parser = argparse.ArgumentParser(description="Generate Excel files from entity JSONs in experiment folder.")
parser.add_argument("experiment_folder", type=str, help="Path to the experiment folder (e.g., lm_studio/outputs/experiment_3)")
args = parser.parse_args()

experiment_folder = args.experiment_folder

# Check if experiment folder exists
if not os.path.isdir(experiment_folder):
    print(f"Experiment folder not found: {experiment_folder}")
    exit(1)

# Loop through each model subfolder
for model_name in os.listdir(experiment_folder):
    model_path = os.path.join(experiment_folder, model_name)
    if not os.path.isdir(model_path):
        continue

    # Prepare output file name
    output_file = os.path.join(experiment_folder, f"entity_extraction_results_{model_name}.xlsx")

    # Get all entity types across files
    all_entity_types = set()
    document_entities = {}

    # Process each JSON file in the model subfolder
    for filename in os.listdir(model_path):
        if filename.endswith("_entities.json"):
            doc_name = filename.replace("_entities.json", "")
            file_path = os.path.join(model_path, filename)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    # Create a dict to hold entities by type for this document
                    entities_by_type = defaultdict(list)

                    # Check if the data has an 'entities' key (as in the example)
                    if 'entities' in data and isinstance(data['entities'], list):
                        for entity_item in data['entities']:
                            if 'text' in entity_item and 'label' in entity_item:
                                entity_type = entity_item['label']
                                entity_text = entity_item['text']

                                all_entity_types.add(entity_type)
                                entities_by_type[entity_type].append(entity_text)

                    document_entities[doc_name] = entities_by_type
                    print(f"[{model_name}] Processed {filename}: found {sum(len(entities) for entities in entities_by_type.values())} entities")

            except Exception as e:
                print(f"[{model_name}] Error processing {filename}: {e}")

    # Create a DataFrame for Excel output
    rows = []
    for doc_name, entities_by_type in document_entities.items():
        # Get the maximum number of entities in any category for this document
        max_entities = max([len(entities) for entities in entities_by_type.values()], default=0)

        # Create rows for this document
        for i in range(max_entities):
            row = {'Document': doc_name}

            for entity_type in all_entity_types:
                entities = entities_by_type.get(entity_type, [])
                row[entity_type] = entities[i] if i < len(entities) else ""

            rows.append(row)

        # Add a blank row after each document
        rows.append({col: "" for col in ['Document'] + list(all_entity_types)})

    # Create DataFrame and export to Excel
    df = pd.DataFrame(rows)
    if not rows:
        print(f"[{model_name}] No data was processed. Check JSON file paths and formats.")
    else:
        df.to_excel(output_file, index=False)
        print(f"[{model_name}] Created {output_file} with columns: Document, {', '.join(all_entity_types)}")

print("\nAll model subfolders processed.")