# 3MF-MDA

3MF-MDA: A macro-meso-micro fusion model with shared-private cross-attention for microRNA-drug association prediction.The model integrates various data sources, including drug SMILES sequences, drug molecular graphs, miRNA sequences, miRNA gip matrix Bipartite Graph and miRNA-drug interactions.


## Key Dependencies

Below are the main libraries and their versions required to run this project. Please fill in the specific versions you are using.

*   **Python**: `3.10`
*   **Torch**: `2.10.0＋cu128`
*   **torch-geometric**: `2.5.3`
*   **Pandas**: `3.0.1`
*   **Numpy**: `1.26.1`

## Usage
 **Run the training script:**
    ```bash
    python training.py
    ```

## Project Structure

- `dataprocess.py`: Script for data loading, preprocessing, and feature engineering.
- `model.py`: Contains the PyTorch implementation of the 3MFModel.
- `training.py`: Main script to train and evaluate the model.
- `utils.py`: Utility functions used across the project.
- `data/`: Directory for storing all raw and processed data files.
- `Predataprocess/`: Scripts and data related to initial feature processing steps.
