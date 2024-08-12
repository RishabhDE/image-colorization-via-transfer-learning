import json

# Load the notebook
with open('Image_colorization.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Collect all imports
imports = set()
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        for line in cell['source']:
            if line.startswith('import') or line.startswith('from'):
                imports.add(line.strip())

# Print all imports
for imp in sorted(imports):
    print(imp)