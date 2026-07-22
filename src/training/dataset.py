import anndata as ad
from pathlib import Path
from data_operations import pretrain_batch_from_x_tf
from tqdm import tqdm
import warnings


class Dataset:
    def __init__(self, source_dir, backed_r=True, test_ratio=0.0, limit_loaded_shards=None):
        backed = 'r' if backed_r else None

        source_dir = Path(source_dir)
        source_fnames = [fn for fn in source_dir.glob('*.h5ad')]
        source_fnames.sort()

        tot_files = len(source_fnames)
        if limit_loaded_shards is not None:
            assert type(limit_loaded_shards) == int
            tot_files = limit_loaded_shards

        train_test_split_i = int(round(tot_files * (1 - test_ratio)))

        self.shards = {
            'train': [],
            'test': []
        }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Observation names are not unique")

            for fi, f in enumerate(tqdm(source_fnames[:tot_files])):
                ds = 'train' if fi < train_test_split_i else 'test'
                try:
                    self.shards[ds].append(ad.read_h5ad(source_dir / f, backed=backed))
                except Exception as e:
                    print(f"{f} file failed to load")
                    print(f"Exception: {e}")
                    

                if limit_loaded_shards is not None and fi > limit_loaded_shards:
                    break
        

    def pretrain_dg(self, batch_size, ds, max_genes=60_000):

        ds_shards = self.shards[ds]
        tot_shards = len(ds_shards)

        this_shard_i = 0
        this_cell_i = 0

        while True:

            this_shard = ds_shards[this_shard_i]

            this_batch = this_shard[this_cell_i:this_cell_i+batch_size]
            this_cell_i += batch_size

            if this_batch.shape[0] < batch_size:
                # handle if the current shard run out of cells
                this_shard_i += 1
                this_shard_i = this_shard_i % tot_shards
                this_cell_i = 0

                this_shard = ds_shards[this_shard_i]

                n_cells_to_add = batch_size - this_batch.shape[0]

                add_to_batch = this_shard[this_cell_i:this_cell_i+n_cells_to_add]
                this_cell_i += n_cells_to_add

                this_batch = ad.concat([this_batch, add_to_batch])


            x_batch = this_batch.X[:, :max_genes]

            input_x, y = pretrain_batch_from_x_tf(x_batch)

            yield input_x, y


if __name__ == '__main__':
    from constants import DATA_ROOT
    dat = Dataset(source_dir=DATA_ROOT / 'shuffled_shards', limit_loaded_shards=1, backed_r=True)
    dg = dat.pretrain_dg(batch_size=1000, ds='train', max_genes=60_000)

    a = next(dg)
    
    for i in tqdm(range(10_000)):
        a = next(dg)

