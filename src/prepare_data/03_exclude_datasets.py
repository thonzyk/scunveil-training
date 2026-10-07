import shutil

from constants import DATA_ROOT, EXCLUDED_DATASET_IDS

ROOT = DATA_ROOT / 'individual_datasets'


def main():
    exclude_dir = ROOT / '_exclude'
    exclude_dir.mkdir(parents=True, exist_ok=True)

    for fname in EXCLUDED_DATASET_IDS:
        orig_file = ROOT / f'{fname}.h5ad'
        if orig_file.exists():
            shutil.move(orig_file, exclude_dir / orig_file.name)
        else:
            print(f"Warning: not found, skipping: {orig_file}")


if __name__ == "__main__":
    main()
