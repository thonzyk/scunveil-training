from constants import DATA_ROOT
import shutil

ROOT = DATA_ROOT / 'individual_datasets'
LIST_OF_EXCLUSIONS = ['ed5d841d-6346-47d4-ab2f-7119ad7e3a35']

def main():
    exclude_dir = ROOT / '_exclude'
    exclude_dir.mkdir(parents=True, exist_ok=True)

    for fname in LIST_OF_EXCLUSIONS:
        orig_file = ROOT / f'{fname}.h5ad'
        if orig_file.exists():
            shutil.move(orig_file, exclude_dir / orig_file.name)
        else:
            print(f"Warning: not found, skipping: {orig_file}")


if __name__ == "__main__":
    main()