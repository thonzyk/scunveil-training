import cellxgene_census
import h5py
import numpy as np
import os
from pathlib import Path

from constants import DATA_ROOT, CENSUS_VERSION, EXCLUDED_DATASET_IDS

OUTPUT_ROOT = Path(os.environ.get('SCUNVEIL_METADATA_OUTPUT_ROOT', DATA_ROOT))

census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

ds_info = next(census['census_info']['datasets'].read()).to_pandas()

shards = sorted((DATA_ROOT / 'shuffled_shards').glob('*.h5ad'))
if not shards:
    raise FileNotFoundError('No training shards found')

used_datasets = set()
for shard in shards:
    with h5py.File(shard, 'r') as source:
        dataset_ids = source['obs/dataset_id']
        categories = dataset_ids['categories'][:]
        for code in np.unique(dataset_ids['codes'][:]):
            if code >= 0:
                value = categories[code]
                used_datasets.add(value.decode() if isinstance(value, bytes) else str(value))

if used_datasets & EXCLUDED_DATASET_IDS:
    raise RuntimeError('Excluded datasets are present in the training shards')

ds_info_filtered = ds_info[ds_info['dataset_id'].isin(used_datasets)]
if set(ds_info_filtered['dataset_id']) != used_datasets:
    raise RuntimeError('Some training shard datasets are missing from Census metadata')

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
ds_info_filtered.to_csv(OUTPUT_ROOT / 'ds_info.csv', index=False)

Supplementary_Table_S1_training_datasets = ds_info_filtered[['dataset_id', 'citation']]
Supplementary_Table_S1_training_datasets.to_csv(OUTPUT_ROOT / 'Supplementary_Table_S1_training_datasets.csv', index=False)
