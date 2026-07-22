import anndata as ad
from constants import DATA_ROOT
import numpy as np

# --- Configuration ---
INPUT_DIR = DATA_ROOT / 'individual_datasets'


def main():
    genes_argsort = np.load(DATA_ROOT / 'genes_argsort.npy')

    # 1. Get list of files
    input_files = list(INPUT_DIR.glob('*.h5ad'))
    input_files = [fn for fn in input_files if not str(fn).endswith('-uploading.h5ad')]

    if not input_files:
        print("No .h5ad files found to process.")
        return

    input_file = input_files[0]
    adata = ad.read_h5ad(input_file, backed='r')

    adata = adata[:1].to_memory()

    adata = adata[:, genes_argsort]

    adata.var.to_csv(DATA_ROOT / 'var_sorted.csv')

if __name__ == "__main__":
    main()
