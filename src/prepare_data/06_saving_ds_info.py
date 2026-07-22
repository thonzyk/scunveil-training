import cellxgene_census
from constants import DATA_ROOT, CENSUS_VERSION

census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

ds_info = next(census['census_info']['datasets'].read()).to_pandas()

ds_info_filtered = ds_info.copy()

used_datasets = []
with open(DATA_ROOT / 'dataset_list.tsv', 'r', encoding='utf-8') as fr:
    for line in fr:
        used_datasets.append(line.strip())

used_datasets = set(used_datasets)

ds_info_filtered = ds_info_filtered[ds_info_filtered["dataset_id"].isin(used_datasets)]

ds_info_filtered.to_csv(DATA_ROOT / 'ds_info.csv', index=False)

Supplementary_Table_S1_training_datasets = ds_info_filtered[['dataset_id', 'citation']]
Supplementary_Table_S1_training_datasets.to_csv(DATA_ROOT / 'Supplementary_Table_S1_training_datasets.csv', index=False)
