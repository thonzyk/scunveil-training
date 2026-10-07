from pathlib import Path


DEBUG = False # DANGER ZONE IF True

CENSUS_VERSION = '2025-11-08'
EXCLUDED_DATASET_IDS = {'ed5d841d-6346-47d4-ab2f-7119ad7e3a35'}

DATA_ROOT = Path('/workspace/data')
RESULTS_ROOT = Path('/workspace/results')

MIN_GENES = 300
MAX_UMIS = 40_000
