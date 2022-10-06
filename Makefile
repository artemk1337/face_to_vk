all:
	@echo "Rules:\n\
 - build - build prj\n\
 - venv - create venv\n\
 - clean - delete created files"

build: venv
	cd docker && sudo docker-compose up -d  # or use sudo docker-compose -f docker/docker-compose.yaml ... up -d
	# @mkdir -p data


venv:
	@conda create --prefix ./venv python=3.9 -y
	@venv/bin/pip install -r requirements.txt

clean:
	# rm -rf tmp
	rm -rf venv
