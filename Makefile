all:
	@echo "Rules:\n\
 - build - build prj\n\
 - venv - create venv\n\
 - clean - delete created files"

# pip list --format=freeze > requirements.txt

tf_conda_components:
    @conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    @mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    @echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

venv: tf_conda_components
	@conda create --prefix ./venv python=3.8 -y
	@venv/bin/pip install -r requirements.txt

build: venv
	cd docker && sudo docker compose up -d  # or use sudo docker compose -f docker/docker-compose.yaml ... up -d

clean:
	rm -rf venv
