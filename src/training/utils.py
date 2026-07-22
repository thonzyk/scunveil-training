from constants import RESULTS_ROOT
import os
import json
import shutil
from datetime import datetime


def create_new_experiment(config):
    experiment_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    assert not os.path.exists(RESULTS_ROOT / 'training' / experiment_id), f'experiment {experiment_id} already exists'
    
    os.mkdir(RESULTS_ROOT / 'training' / experiment_id)
    os.mkdir(RESULTS_ROOT / 'training' / experiment_id / 'weights')

    with open(RESULTS_ROOT / 'training' / experiment_id / 'config.json', 'w', encoding='utf-8') as fw:
        json.dump(config, fw, indent=4)

    f_names = os.listdir('/')
    f_names = [f_name for f_name in f_names if str(f_name).endswith('.py')]

    for f_name in f_names:
        try:
            shutil.copy(f_name, RESULTS_ROOT / 'training' / experiment_id / 'src' / f_name)
        except Exception as e:
            print(f"Error occurred: {e}")

    return experiment_id
