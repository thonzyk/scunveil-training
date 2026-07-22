import anndata as ad
from constants import DATA_ROOT, DEBUG
import numpy as np

from tqdm import tqdm

# --- Configuration ---
INPUT_DIR = DATA_ROOT / 'individual_datasets'
OUTPUT_DIR = DATA_ROOT / 'shuffled_shards'

N_SHARDS = 1000
if DEBUG:
    N_SHARDS = N_SHARDS // 10

def main():
    genes_argsort = np.load(DATA_ROOT / 'genes_argsort.npy')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get list of files
    input_files = list(INPUT_DIR.glob('*.h5ad'))
    input_files = [fn for fn in input_files if not str(fn).endswith('-uploading.h5ad')]

    if not input_files:
        print("No .h5ad files found to process.")
        return
    
    if DEBUG:
        input_files = input_files[:2]

    ds_indices = []
    cell_indices = []
    total_size = 0
    ref_var_names = None
    full_var = None
    all_datasets_h5ads = []

    print('Scanning datasets (backed=r)...')
    for ds_i, file in enumerate(tqdm(input_files)):
        this_h5ad = ad.read_h5ad(file, backed='r')

        if ref_var_names is None:
            ref_var_names = this_h5ad.var_names.to_numpy().copy()
            full_var = this_h5ad.var.copy()
        else:
            assert this_h5ad.n_vars == ref_var_names.shape[0]
            assert np.array_equal(this_h5ad.var_names.to_numpy(), ref_var_names)

        total_size += this_h5ad.n_obs
        ds_indices.append(ds_i * np.ones((this_h5ad.n_obs,), dtype='int64'))
        cell_indices.append(np.arange(this_h5ad.n_obs))

        all_datasets_h5ads.append(this_h5ad)

    ds_indices = np.concatenate(ds_indices)
    cell_indices = np.concatenate(cell_indices)

    print(f'Loaded {total_size} cells.')

    shard_size = int(np.floor(total_size / N_SHARDS))
    print(f'{N_SHARDS} shards of size {shard_size} will be generated.')
    remaining_cells = total_size - shard_size * N_SHARDS
    print(f'{remaining_cells} cells will be discarded in order to have equaly large shards.')

    print('Sharding started...')

    permutation = np.random.permutation(total_size)[:shard_size * N_SHARDS]

    a_range = range(N_SHARDS)
    if DEBUG:
        a_range = range(min(2, N_SHARDS))

    for shard_i in tqdm(a_range):
        select = permutation[shard_i * shard_size:(shard_i + 1) * shard_size]
        this_ds_indices = ds_indices[select]
        this_cell_indices = cell_indices[select]

        unique_ds_indices = np.unique(this_ds_indices)

        shard = []

        for ds_i in unique_ds_indices:
            select = this_ds_indices == ds_i

            ds = all_datasets_h5ads[ds_i]

            this_append = ds[this_cell_indices[select]]
            shard.append(this_append.to_memory())

        shard = ad.concat(shard)
        
        shard.var = full_var

        shard = shard[np.random.permutation(shard_size)]
        shard = shard[:, genes_argsort]

        shard.obs = shard.obs[['dataset_id', 'assay', 'cell_type', 'development_stage', 'disease', 'donor_id', 'self_reported_ethnicity', 'sex', 'suspension_type', 'tissue', 'tissue_type', 'tissue_general']]

        out_name = str(shard_i).zfill(len(str(N_SHARDS))) + '.h5ad'
        shard.write(OUTPUT_DIR  / out_name, compression="gzip")


if __name__ == "__main__":
    main()
