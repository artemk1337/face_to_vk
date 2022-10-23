# Face to vk

---

## Requirement components

- Ubuntu 20.04 LTS or later
- Anaconda
- Docker & docker-compose
- Postgresql 13 or later

## Deploy

### Build project

Use **Makefile**:

```bash
make build
```

### Preparing CUDA for ML (only NVidia GPUs)

Install cudnn on Ubuntu:

```bash
sudo apt install nvidia-cudnn
```

Install other nvidia components for Tensorflow:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```
