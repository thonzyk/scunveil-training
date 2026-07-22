FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

RUN apt-get update && apt-get install -y git libcairo2 && rm -rf /var/lib/apt/lists/*

# (Optional) keep pip updated, but the key is constraints
RUN python -m pip install --upgrade pip

# Freeze the fragile ABI bits (edit as needed)
RUN printf "%s\n" \
  "numpy<2" \
  > /tmp/constraints.txt

# Install everything under constraints
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
      pandas matplotlib tqdm wandb anndata cellxgene-census scikit-learn
