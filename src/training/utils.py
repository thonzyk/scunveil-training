from constants import RESULTS_ROOT
import os
import json
import shutil
from datetime import datetime
from pathlib import Path


def create_new_experiment(config):
    experiment_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    assert not os.path.exists(RESULTS_ROOT / 'training' / experiment_id), f'experiment {experiment_id} already exists'
    
    os.mkdir(RESULTS_ROOT / 'training' / experiment_id)
    os.mkdir(RESULTS_ROOT / 'training' / experiment_id / 'weights')

    with open(RESULTS_ROOT / 'training' / experiment_id / 'config.json', 'w', encoding='utf-8') as fw:
        json.dump(config, fw, indent=4)

    source_dir = Path(__file__).resolve().parent
    backup_dir = RESULTS_ROOT / 'training' / experiment_id / 'src'
    backup_dir.mkdir()

    for source_file in source_dir.glob('*.py'):
        try:
            shutil.copy2(source_file, backup_dir / source_file.name)
        except Exception as e:
            print(f"Error occurred: {e}")

    return experiment_id
