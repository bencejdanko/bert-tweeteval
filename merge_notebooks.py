import json
import os

def merge_notebooks(filenames, script_dir, output_filename):
    merged_notebook = None
    
    # 1. Start with the first notebook as a template for metadata
    if filenames and os.path.exists(filenames[0]):
        with open(filenames[0], 'r', encoding='utf-8') as f:
            merged_notebook = json.load(f)
            merged_notebook['cells'] = [] # Start fresh
    else:
        # Fallback if first notebook is missing
        merged_notebook = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4
        }

    # 2. Add source scripts at the top
    if os.path.isdir(script_dir):
        # Sort scripts for consistent order (alphabetical, excluding __init__.py)
        scripts = sorted([f for f in os.listdir(script_dir) if f.endswith('.py') and f != '__init__.py'])
        for script_name in scripts:
            script_path = os.path.join(script_dir, script_name)
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a markdown cell followed by a code cell for each script
            merged_notebook['cells'].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### Source: `{script_name}`\n"]
            })
            merged_notebook['cells'].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [content]
            })

    # 3. Append all notebooks
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping.")
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        merged_notebook['cells'].extend(nb['cells'])
            
    if merged_notebook:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(merged_notebook, f, indent=1, ensure_ascii=False)
        print(f"Successfully merged source scripts and {len(filenames)} notebooks into {output_filename}")
    else:
        print("No notebooks were merged.")

if __name__ == "__main__":
    order = [
        "bert-tweeteval-baseline.ipynb",
        "bert-tweeteval-train.ipynb",
        "bert-tweeteval-corruption.ipynb",
        "bert-tweeteval-error-analysis.ipynb",
        "llm-evaluation.ipynb",
        "n-gram-modeling.ipynb"
    ]
    merge_notebooks(order, "src", "full-notebook.ipynb")
