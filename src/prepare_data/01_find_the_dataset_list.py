import cellxgene_census
from constants import DATA_ROOT, CENSUS_VERSION, DEBUG

# Open the census
census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

# Access the Human observation (cell) metadata directly
# Path: census_data -> homo_sapiens -> obs
human_obs = census["census_data"]["homo_sapiens"].obs

print("Reading obs...")

# Read ONLY the 'donor_id' column for all cells
# This returns a PyArrow table which is extremely fast
query = human_obs.read(column_names=["dataset_id", "is_primary_data"])

# Convert to pandas and get unique values
obs_df = query.concat().to_pandas()
obs_df = obs_df[obs_df["is_primary_data"]]
dataset_list = obs_df["dataset_id"].unique().tolist()

if DEBUG:
    import numpy as np
    dataset_list = list(np.random.choice(dataset_list, size=2))


print(f"Found {len(dataset_list)} unique datasets.")

# Write to file
with open(DATA_ROOT / 'dataset_list.tsv', 'w', encoding='utf-8') as fw:
    for ds_id in dataset_list:
        print(ds_id, file=fw)

census.close()