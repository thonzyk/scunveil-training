import cellxgene_census
import anndata as ad
import os
import gc
import time  # Added time module
import random
import numpy as np
from constants import DATA_ROOT, CENSUS_VERSION, MAX_UMIS, MIN_GENES

ROOT = DATA_ROOT / 'individual_datasets'
dataset_list_path = DATA_ROOT / 'dataset_list.tsv'

def format_time(seconds):
    """Helper to format seconds into H:M:S"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    # 1. Setup paths
    ROOT.mkdir(parents=True, exist_ok=True)

    trash_dir = ROOT / '_trash'
    trash_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Read the dataset list
    with open(dataset_list_path, 'r') as f:
        all_datasets = [line.strip() for line in f if line.strip()]

    # shuffling of names ensures unbiased estimates of remaning time
    random.shuffle(all_datasets)
        
    print(f"Total datasets to process: {len(all_datasets)}")

    # Delete all non finished uploads
    for file in ROOT.glob('*-uploading.h5ad'):
        file.unlink()

    # 3. Check which ones are already done
    existing_files = set(os.listdir(ROOT)).union(set(os.listdir(ROOT / '_trash')))
    datasets_to_process = [d for d in all_datasets if f"{d}.h5ad" not in existing_files]

    total_count = len(datasets_to_process)
    skipped_count = len(all_datasets) - total_count

    print(f"Skipping {skipped_count} already processed.")
    print(f"Starting download for {total_count} datasets...")

    # 4. Open the Census
    census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

    # Track start time
    start_time = time.time()

    # 5. Loop inside Python
    for i, dataset_id in enumerate(datasets_to_process, 1):
        try:
            # --- Processing Start ---
            print(f'Processing {dataset_id}')

            obs_filter = (
                f"dataset_id == '{dataset_id}' "
                f"and is_primary_data == True "
                f"and raw_sum < {MAX_UMIS} "
                f"and nnz > {MIN_GENES} "
            )
            
            adata = cellxgene_census.get_anndata(
                census=census,
                organism="Homo sapiens",
                obs_value_filter=obs_filter,
                measurement_name='RNA'
            )

            assert np.sum(adata.X.data - np.round(adata.X.data)) < 1e-2, 'the data are not round numbers!'

            # casting to correct type
            adata.X.data = adata.X.data.astype('uint16')
            adata.X.indices = adata.X.indices.astype('uint16')
            adata.X.indptr = adata.X.indptr.astype('int64')

            save_path = ROOT / f"{dataset_id}-uploading.h5ad"
            adata.write(save_path, compression="gzip")

            os.rename(ROOT / f"{dataset_id}-uploading.h5ad", ROOT / f"{dataset_id}.h5ad")

            assert adata.shape[0] == int(adata.n_obs)
            n_this_obs = int(adata.n_obs)

            del adata
            gc.collect()

            # Trahs empty datasets
            if n_this_obs == 0:
                file_to_trash = ROOT / f"{dataset_id}.h5ad"
                file_to_trash.rename(ROOT / '_trash' / f"{dataset_id}.h5ad")

            # --- Processing End ---

            # Custom Progress Logic
            elapsed_time = time.time() - start_time
            proportion_done = i / total_count
            
            # Estimate total time based on current pace, then subtract elapsed
            estimated_total_time = elapsed_time / proportion_done
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"[{i}/{total_count}] Finished {dataset_id}. "
                  f"Elapsed: {format_time(elapsed_time)} | "
                  f"Remaining: {format_time(remaining_time)}")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to download {dataset_id}: {e}")
            continue

    census.close()
    print("Done!")

if __name__ == "__main__":
    main()