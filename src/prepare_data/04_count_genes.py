import anndata as ad
from constants import DATA_ROOT, DEBUG
import numpy as np

from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = DATA_ROOT / 'individual_datasets'

BATCH_SIZE = 100_000
if DEBUG:
    BATCH_SIZE = BATCH_SIZE // 100

def main():

    # 1. Get list of files
    input_files = list(INPUT_DIR.glob('*.h5ad'))
    input_files = [fn for fn in input_files if not str(fn).endswith('-uploading.h5ad')]

    if not input_files:
        print("No .h5ad files found to process.")
        return
    
    if DEBUG:
        input_files = input_files[:2]

    umi_nonzero = None
    ref_var_names = None
    gene_names = None

    for file in tqdm(input_files):
        this_h5ad = ad.read_h5ad(file, backed='r')
        try:
            # compatibility assertions (once we have a reference)
            if ref_var_names is None:
                ref_var_names = this_h5ad.var_names.to_numpy().copy()
            else:
                assert this_h5ad.n_vars == ref_var_names.shape[0]
                assert np.array_equal(this_h5ad.var_names.to_numpy(), ref_var_names)

            for i in range(0, this_h5ad.n_obs, BATCH_SIZE):
                batch = this_h5ad[i:i+BATCH_SIZE].to_memory()

                if umi_nonzero is None:
                    umi_nonzero = np.zeros((this_h5ad.n_vars,), dtype='int64')

                x = batch.X
                # faster nonzero counting for sparse matrices
                umi_nonzero += x.getnnz(axis=0).astype('int64')

        finally:
            # close the backed file handle
            gene_names = np.array(list(this_h5ad.var['feature_name']))
            this_h5ad.file.close()

    arg_sort = np.argsort(-umi_nonzero)

    np.save(DATA_ROOT / 'umi_nonzero.npy', umi_nonzero)
    np.save(DATA_ROOT / 'genes_argsort.npy', arg_sort)

    print('Top expressed genes sanity check:', ", ".join(gene_names[arg_sort[:50]]))
    print('Done!')


if __name__ == "__main__":
    main()
