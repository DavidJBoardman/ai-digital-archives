import pandas as pd
import os
import numpy as np
import re
import argparse

def load_excel_data(filepath):
    try:
        return pd.read_excel(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def normalize_text(text):
    if pd.isna(text) or text == "":
        return ""
    # Convert to lowercase and remove extra whitespace
    return str(text).lower().strip()

def calculate_entity_metrics(ground_truth, predictions, show_examples=False, max_examples=5):
    # Filter out None/NaN values
    valid_indices = [i for i, (gt, pred) in enumerate(zip(ground_truth, predictions)) 
                     if not (pd.isna(gt) and pd.isna(pred))]
    
    if not valid_indices:
        return {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "num_samples": 0
        }
    
    gt_filtered = [normalize_text(ground_truth[i]) for i in valid_indices]
    pred_filtered = [normalize_text(predictions[i]) for i in valid_indices]
    
    # Display examples if requested
    if show_examples and valid_indices:
        print("\n  Example Comparisons:")
        displayed = 0
        for i, (gt, pred) in enumerate(zip(gt_filtered, pred_filtered)):
            if displayed >= max_examples:
                break
            if gt or pred:  # Only show if at least one has content
                match_status = "✓" if gt == pred else "✗"
                print(f"    {match_status} GT: '{gt}' | Pred: '{pred}'")
                displayed += 1
    
    # Calculate exact match
    exact_matches = sum(1 for gt, pred in zip(gt_filtered, pred_filtered) if gt == pred)
    exact_match_ratio = exact_matches / len(valid_indices) if valid_indices else 0
    
    # For NER evaluation:
    # True Positive: Predicted entity matches ground truth
    # False Positive: Predicted entity where there should be none, or wrong entity
    # False Negative: Failed to predict an entity that exists in ground truth
    
    # Count entities
    true_pos = sum(1 for gt, pred in zip(gt_filtered, pred_filtered) 
                    if gt != "" and pred != "" and gt == pred)
    
    false_pos = sum(1 for gt, pred in zip(gt_filtered, pred_filtered) 
                     if (gt == "" and pred != "") or (gt != "" and pred != "" and gt != pred))
    
    false_neg = sum(1 for gt, pred in zip(gt_filtered, pred_filtered) 
                     if gt != "" and (pred == "" or gt != pred))
    
    # Calculate precision, recall, F1
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "exact_match": exact_match_ratio,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "num_samples": len(valid_indices),
        "true_positives": true_pos,
        "false_positives": false_pos,
        "false_negatives": false_neg
    }

def compare_models_to_ground_truth(model_files, ground_truth_file, show_examples=False):
    # Load ground truth data
    ground_truth_df = load_excel_data(ground_truth_file)
    if ground_truth_df is None:
        print(f"Error: Could not load ground truth file {ground_truth_file}")
        return
    
    print(f"Loaded ground truth data with {len(ground_truth_df)} rows")
    print(f"Ground truth columns: {ground_truth_df.columns.tolist()}")
    
    # Load model data
    model_data = {}
    for model, filepath in model_files.items():
        df = load_excel_data(filepath)
        if df is not None:
            model_data[model] = df
            print(f"Loaded {model} data with {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")
    
    # Find common entity types between ground truth and models
    gt_columns = ground_truth_df.columns.tolist()
    entity_types = [col for col in gt_columns if col != 'Document']  # Exclude the document ID column
    
    print(f"\nEntity types for evaluation: {entity_types}")
    
    # Results dictionary to store metrics for all models and entity types
    results = {}
    
    # Evaluate each model against ground truth for each entity type
    for entity_type in entity_types:
        print(f"\n--- Evaluating entity type: {entity_type} ---")
        entity_results = {}
        
        # Ground truth data for this entity type
        ground_truth = ground_truth_df[entity_type].values
        
        for model, df in model_data.items():
            # Check if the model has this entity type
            if entity_type in df.columns:
                model_predictions = df[entity_type].values
                
                # Ensure we only compare rows that exist in both datasets
                min_len = min(len(ground_truth), len(model_predictions))
                gt_data = ground_truth[:min_len]
                pred_data = model_predictions[:min_len]
                
                # Calculate metrics
                metrics = calculate_entity_metrics(gt_data, pred_data, show_examples)
                entity_results[model] = metrics
                
                # Print results
                print(f"{model} - {entity_type}:")
                print(f"  Exact Match: {metrics['exact_match']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
                print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
                print(f"  Number of samples: {metrics['num_samples']}")
            else:
                print(f"  Entity type {entity_type} not found in model {model}. Skipping.")
        
        results[entity_type] = entity_results
    
    # Overall model comparison
    print("\n=== Overall Model Comparison ===")
    
    overall_metrics = {model: {
        "f1": [], "precision": [], "recall": [], "exact_match": [], 
        "sample_count": 0, "tp": 0, "fp": 0, "fn": 0
    } for model in model_data.keys()}
    
    for entity_type, model_results in results.items():
        for model, metrics in model_results.items():
            if metrics["num_samples"] > 0:
                overall_metrics[model]["f1"].append(metrics["f1_score"])
                overall_metrics[model]["precision"].append(metrics["precision"])
                overall_metrics[model]["recall"].append(metrics["recall"])
                overall_metrics[model]["exact_match"].append(metrics["exact_match"])
                overall_metrics[model]["sample_count"] += metrics["num_samples"]
                overall_metrics[model]["tp"] += metrics["true_positives"]
                overall_metrics[model]["fp"] += metrics["false_positives"]
                overall_metrics[model]["fn"] += metrics["false_negatives"]
    
    # Print overall metrics for each model
    for model, metrics in overall_metrics.items():
        if metrics["sample_count"] > 0:
            # Calculate macro-average (average of metrics across all entity types)
            macro_f1 = np.mean(metrics["f1"])
            macro_precision = np.mean(metrics["precision"])
            macro_recall = np.mean(metrics["recall"])
            macro_exact_match = np.mean(metrics["exact_match"])
            
            # Calculate micro-average (aggregate TP, FP, FN across all entity types)
            micro_precision = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0
            micro_recall = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            
            print(f"\n{model.upper()}:")
            print(f"  Macro-Avg F1: {macro_f1:.4f}")
            print(f"  Macro-Avg Precision: {macro_precision:.4f}")
            print(f"  Macro-Avg Recall: {macro_recall:.4f}")
            print(f"  Micro-Avg F1: {micro_f1:.4f}")
            print(f"  Micro-Avg Precision: {micro_precision:.4f}")
            print(f"  Micro-Avg Recall: {micro_recall:.4f}")
            print(f"  Avg Exact Match: {macro_exact_match:.4f}")
            print(f"  Total Samples: {metrics['sample_count']}")
            print(f"  Total TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

def main():
    parser = argparse.ArgumentParser(description='Compare NER results across different models against a ground truth')
    parser.add_argument('--ground-truth', type=str, 
                        help='Path to the ground truth Excel file')
    parser.add_argument('--show-examples', action='store_true', 
                        help='Show examples of ground truth vs. predictions')
    args = parser.parse_args()
    
    # Default ground truth file
    default_ground_truth = "datasets/labelled_data/ground_truth.xlsx"
    ground_truth_file = args.ground_truth if args.ground_truth else default_ground_truth
    
    # Model output files
    base_dir = "lm_studio/outputs/experiment_2"
    model_files = {
        "llama4": os.path.join(base_dir, "entity_extraction_results_llama4.xlsx"),
        "llama3": os.path.join(base_dir, "entity_extraction_results_llama3.xlsx"),
        "gemma": os.path.join(base_dir, "entity_extraction_results_gemma.xlsx")
    }
    
    compare_models_to_ground_truth(model_files, ground_truth_file, args.show_examples)

if __name__ == "__main__":
    main()
