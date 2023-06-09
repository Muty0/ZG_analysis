# ZG_analysis

This repository contains the scripts and files for ZG analysis. 

## Files

- **main.ipynb**: This Jupyter Notebook contains the main script for the analysis.
- **input_file.py**: This Python script contains the input file configurations.
- **selection_cut.py**: This Python script contains the event selection criteria.

## Usage
### preprocessing: 
- Please check this page from Jing : https://gitee.com/jinggiteeee/nanoaod_framework
- Follow Jing's instruction, you can get all the preprocessed root files.
- Use [realpath -s *.root > path.txt] command to get your own root file path, and replace them in the input_file.py later. 

### Selection and analize
1. Clone the repository:
git clone https://github.com/Muty0/ZG_analysis.git

2. Open and run the `main.ipynb` Jupyter Notebook for the analysis.

3. The input files are loaded and configured using `input_file.py`.

4. Event selection criteria are applied using `selection_cut.py`.

## Dependencies

- Python 3.x
- Jupyter Notebook / VSCode
- (List any other required libraries or packages)


