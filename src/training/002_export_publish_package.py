"""Build inference artifacts from one training checkpoint."""

from pathlib import Path
import json
import shutil
import tempfile

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from constants import DATA_ROOT
from data_operations import simple_scipy_norm_x
from dataset import Dataset
from model import RNABagModel, c_xent


WEIGHTS_PATH = Path('/workspace/results/training/2026-10-02_11-54-21_new_main/weights/0064.weights.h5')
N_LOSS_CELLS = 100_000
N_PCA_CELLS = 1_000_000
BATCH_SIZE = 1_000
MAX_LOSS_INCREASE = 0.001  # nats per cell


def export_float16(source_path, output_path):
    """Copy model weights in bounded chunks; omit optimizer state."""
    with h5py.File(source_path, 'r') as source, h5py.File(output_path, 'x') as output:
        total_bytes = 0

        def count_float_bytes(_, item):
            nonlocal total_bytes
            if isinstance(item, h5py.Dataset) and item.dtype.kind == 'f':
                total_bytes += item.size * item.dtype.itemsize

        for name, item in source.items():
            if name != 'optimizer':
                item.visititems(count_float_bytes)

        with tqdm(total=total_bytes, unit='B', unit_scale=True,
                  smoothing=0, desc='Exporting float16 weights') as progress:
            def copy_group(src, dst):
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                for name, item in src.items():
                    if isinstance(item, h5py.Group):
                        copy_group(item, dst.create_group(name))
                    elif item.dtype.kind == 'f':
                        target = dst.create_dataset(name, shape=item.shape, dtype='float16')
                        for key, value in item.attrs.items():
                            target.attrs[key] = value
                        if item.ndim == 0:
                            target[()] = item[()]
                            progress.update(item.dtype.itemsize)
                        else:
                            row_size = int(np.prod(item.shape[1:]))
                            step = max(1, 1_000_000 // row_size)
                            for start in range(0, item.shape[0], step):
                                block = item[start:start + step]
                                target[start:start + step] = block
                                progress.update(block.nbytes)
                    else:
                        src.copy(item, dst, name=name)

            for name, item in source.items():
                if name != 'optimizer':
                    copy_group(item, output.create_group(name))
            for key, value in source.attrs.items():
                output.attrs[key] = value


def evaluate_loss(original, rounded, dataset, n_genes):
    """Compare both checkpoints on each identical training-style batch."""
    batches = dataset.pretrain_dg(BATCH_SIZE, 'train', max_genes=n_genes)
    totals = np.zeros(2, dtype=np.float64)
    with tqdm(total=N_LOSS_CELLS, unit='cell', smoothing=0,
              desc='Checking heldout loss') as progress:
        for start in range(0, N_LOSS_CELLS, BATCH_SIZE):
            inputs, targets = next(batches)
            size = min(BATCH_SIZE, N_LOSS_CELLS - start)
            inputs, targets = inputs[:size], targets[:size]
            totals[0] += float(c_xent(targets, original(inputs, training=False))) * size
            totals[1] += float(c_xent(targets, rounded(inputs, training=False))) * size
            progress.update(size)
    return totals / N_LOSS_CELLS


def fit_pca(embedder, dataset, n_genes, emb_dim):
    """Exact full PCA from a streamed, float64 running covariance."""
    if N_PCA_CELLS < max(2, emb_dim):
        raise ValueError('N_PCA_CELLS must be at least max(2, emb_dim)')
    mean = np.zeros(emb_dim, dtype=np.float64)
    m2 = np.zeros((emb_dim, emb_dim), dtype=np.float64)
    n = 0
    with tqdm(total=N_PCA_CELLS, unit='cell', smoothing=0,
              desc='Fitting PCA') as progress:
        for shard in dataset.shards['train']:
            if n >= N_PCA_CELLS:
                break
            for start in range(0, min(shard.n_obs, N_PCA_CELLS - n), BATCH_SIZE):
                stop = min(start + BATCH_SIZE, shard.n_obs, start + N_PCA_CELLS - n)
                counts = shard.X[start:stop, :n_genes]
                inputs = simple_scipy_norm_x(counts)
                embeddings = embedder(inputs, training=False).numpy().astype(np.float64)
                if not np.isfinite(embeddings).all():
                    raise FloatingPointError('Non-finite embeddings')
                batch_n = len(embeddings)
                batch_mean = embeddings.mean(axis=0)
                centered = embeddings - batch_mean
                delta = batch_mean - mean
                m2 += centered.T @ centered
                m2 += np.outer(delta, delta) * (n * batch_n / (n + batch_n))
                mean += delta * (batch_n / (n + batch_n))
                n += batch_n
                progress.update(batch_n)

    covariance = m2 / (n - 1)
    print('Diagonalizing the full covariance matrix...', flush=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1]
    pca_mat = eigenvectors[:, ::-1].astype(np.float32)  # columns are PCs
    pca_mean = mean[None, :].astype(np.float32)

    # This checks the stored matrix orientation, which reconstruction alone cannot.
    k = min(16, emb_dim)
    leading = pca_mat[:, :k].astype(np.float64)
    residual = np.linalg.norm(covariance @ leading - leading * eigenvalues[:k])
    residual /= max(np.linalg.norm(eigenvalues[:k]), 1e-30)
    if not np.isfinite(residual) or residual > 1e-5:
        raise RuntimeError(f'PCA projection orientation check failed: {residual:g}')
    if eigenvalues[-1] < -1e-8 * eigenvalues[0]:
        raise RuntimeError('PCA covariance has a substantially negative eigenvalue')
    total_variance = np.maximum(eigenvalues, 0).sum()
    if total_variance <= 0:
        raise RuntimeError('Embeddings have no variation for PCA')
    explained_512 = np.maximum(eigenvalues[:min(512, emb_dim)], 0).sum() / total_variance
    return pca_mean, pca_mat, float(explained_512), float(residual)


def main():
    if not WEIGHTS_PATH.is_file() or not WEIGHTS_PATH.name.endswith('.weights.h5'):
        raise FileNotFoundError(f'Select an existing .weights.h5 file: {WEIGHTS_PATH}')
    if WEIGHTS_PATH.parent.name != 'weights':
        raise ValueError('Select an original float32 checkpoint from the weights directory')
    experiment_dir = WEIGHTS_PATH.parent.parent
    publish_dir = experiment_dir / 'publish'
    if publish_dir.exists():
        raise FileExistsError(f'Publish directory already exists: {publish_dir}')
    with (experiment_dir / 'config.json').open(encoding='utf-8') as file:
        config = json.load(file)
    n_genes, emb_dim = int(config['n_genes']), int(config['emb_dim'])
    print('Opening backed training shards (metadata only)...', flush=True)
    dataset = Dataset(DATA_ROOT / 'shuffled_shards', backed_r=True)
    available = sum(shard.n_obs for shard in dataset.shards['train'])
    if available < max(N_LOSS_CELLS, N_PCA_CELLS):
        raise ValueError(f'Only {available:,} training cells are available')
    print(f'Source: {WEIGHTS_PATH}', flush=True)
    print(f'Cells: {N_LOSS_CELLS:,} for loss, {N_PCA_CELLS:,} for PCA; batch size: {BATCH_SIZE}', flush=True)
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=experiment_dir) as temp:
        work = Path(temp)
        export_float16(WEIGHTS_PATH, work / 'weights.weights.h5')
        original = RNABagModel(n_vars=n_genes, n_layers=int(config['n_layers']),
                               emb_dim=emb_dim, ff_dim=config.get('ff_dim')).model
        rounded = RNABagModel(n_vars=n_genes, n_layers=int(config['n_layers']),
                              emb_dim=emb_dim, ff_dim=config.get('ff_dim')).model
        original.load_weights(WEIGHTS_PATH)
        rounded.load_weights(work / 'weights.weights.h5')
        original_loss, exported_loss = evaluate_loss(original, rounded, dataset, n_genes)
        if not np.isfinite([original_loss, exported_loss]).all():
            raise FloatingPointError('Non-finite loss during float16 comparison')
        loss_increase = exported_loss - original_loss
        print(f'Loss: float32={original_loss:.6f}, float16={exported_loss:.6f}, '
              f'increase={loss_increase:+.6f} nats/cell', flush=True)
        if loss_increase > MAX_LOSS_INCREASE:
            raise RuntimeError(f'Float16 loss increase exceeds {MAX_LOSS_INCREASE} nats/cell')

        embedder = tf.keras.Model(rounded.input, rounded.get_layer('out_emb').output)
        del original
        pca_mean, pca_mat, explained_512, pca_residual = fit_pca(
            embedder, dataset, n_genes, emb_dim)
        np.save(work / 'pca_mean.npy', pca_mean)
        np.save(work / 'pca_mat.npy', pca_mat)
        shutil.copyfile(experiment_dir / 'config.json', work / 'config.json')
        shutil.copyfile(DATA_ROOT / 'var_sorted.csv', work / 'var_sorted.csv')
        report = {
            'source_weights': str(WEIGHTS_PATH),
            'loss_cells': N_LOSS_CELLS,
            'pca_cells': N_PCA_CELLS,
            'float32_loss': original_loss,
            'float16_loss': exported_loss,
            'loss_increase': loss_increase,
            'explained_variance_first_512': explained_512,
            'pca_eigenvector_residual': pca_residual,
        }
        (work / 'export_report.json').write_text(json.dumps(report, indent=2) + '\n')
        work.rename(publish_dir)

    print(f'PCA variance in first 512 components: {explained_512:.2%}')
    print(f'PCA orientation residual: {pca_residual:.2e}')
    print(f'Publish package ready: {publish_dir}')


if __name__ == '__main__':
    main()
