
import pandas as pd
import json
import os
from app.src.config import DATA_DIR

def extract_category_hierarchy():
    """
    Reads the KICKSTARTER_CLEAN_BASE.csv file, parses the JSON 'category' column,
    and extracts a hierarchy of main categories and sub-categories.
    Saves the result as a JSON file.
    """
    
    # Path to the dataset
    dataset_path = os.path.join(DATA_DIR, 'KICKSTARTER_CLEAN_BASE.csv')
    output_path = os.path.join(DATA_DIR, 'category_hierarchy.json')

    print(f"Reading dataset from: {dataset_path}")
    
    try:
        # Read only the 'category' column to save memory
        df = pd.read_csv(dataset_path, usecols=['category'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    hierarchy = {}

    print("Extracting categories...")
    
    for _idx, row in df.iterrows():
        try:
            cat_data = json.loads(row['category'])
            
            # parent_name is the main category (e.g., "Technology")
            # name is the sub-category (e.g., "Robots")
            
            # Sometimes parent_name might be missing if it's a root category, 
            # but usually sub-categories have a parent.
            
            if 'parent_name' in cat_data:
                main_cat = cat_data['parent_name']
                sub_cat = cat_data['name']
            else:
                # If there is no parent_name, it might be a top-level category itself acting as a sub-category in some contexts,
                # or just a category without a specific sub-category.
                # For this specific dataset, let's treat it as a main category.
                main_cat = cat_data['name']
                sub_cat = None # Or maybe "" or replicate main_cat? Let's check logic.

            if main_cat not in hierarchy:
                hierarchy[main_cat] = set()
            
            if sub_cat:
                hierarchy[main_cat].add(sub_cat)
                
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    # Convert sets to sorted lists for JSON serialization
    sorted_hierarchy = {k: sorted(list(v)) for k, v in hierarchy.items()}
    
    # Sort keys as well
    sorted_hierarchy = dict(sorted(sorted_hierarchy.items()))

    # Calculate stats
    total_main = len(sorted_hierarchy)
    total_sub = sum(len(subs) for subs in sorted_hierarchy.values())
    
    print(f"Extracted {total_main} main categories and {total_sub} sub-categories.")

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(sorted_hierarchy, f, indent=4)
        
    print(f"Category hierarchy saved to: {output_path}")

if __name__ == "__main__":
    # Ensure the parent package 'app' is recognized if run as script
    # This might require checking sys.path or running as module
    # But for simplicity, let's just rely on the imports working if run from root.
    # To run this: python -m app.src.extract_categories
    extract_category_hierarchy()
