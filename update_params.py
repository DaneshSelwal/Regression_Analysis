import json
import os
import glob
import re

def extract_new_params(tuning_nb_path):
    """Extracts best_scores_autosampler dictionary string from the tuning notebook."""
    if not os.path.exists(tuning_nb_path):
        return None
    
    with open(tuning_nb_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {tuning_nb_path}: {e}")
            return None

    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            # Look for the cell that HAS the dictionary in outputs
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'execute_result':
                    data_payload = output.get('data', {})
                    if 'text/plain' in data_payload:
                        text = "".join(data_payload['text/plain'])
                        # The text/plain output should be the dictionary string
                        if text.strip().startswith('{') and "Random Forest" in text:
                            return text
    return None

def update_notebook(target_nb_path, new_params_str):
    """Replaces best_scores_autosampler definition in target notebook."""
    if not os.path.exists(target_nb_path):
        return False
    
    with open(target_nb_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {target_nb_path}: {e}")
            return False

    updated = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source_lines = cell.get('source', [])
            source_text = "".join(source_lines)
            
            if 'best_scores_autosampler = {' in source_text:
                prefix = "best_scores_autosampler = "
                new_source = prefix + new_params_str
                
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                if cell['source'][-1].endswith('\n'):
                     cell['source'][-1] = cell['source'][-1][:-1]
                
                updated = True

    if updated:
        with open(target_nb_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        return True
    return False

def main():
    base_dir = "examples"
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"Found datasets: {datasets}")

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        tuning_nb = os.path.join(dataset_path, "hyperparameter_tuning", "Optuna_autosampler.ipynb")
        
        print(f"\nProcessing dataset: {dataset}")
        new_params = extract_new_params(tuning_nb)
        
        if not new_params:
            print(f"  [SKIP] Could not find tuned parameters dictionary in {tuning_nb}")
            continue
        
        print(f"  [OK] Extracted tuned parameters (length: {len(new_params)})")

        all_nbs = glob.glob(os.path.join(dataset_path, "**", "*.ipynb"), recursive=True)
        target_nbs = [nb for nb in all_nbs if "hyperparameter_tuning" not in nb]

        print(f"  Updating {len(target_nbs)} notebooks...")
        for nb in target_nbs:
            if update_notebook(nb, new_params):
                print(f"    [UPDATED] {nb}")
            else:
                print(f"    [NO-CHANGE] {nb}")

if __name__ == "__main__":
    main()
