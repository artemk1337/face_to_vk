all:
	@echo "Rules:\n\
 - build - build prj\n\
 - venv - create venv\n\
 - clean - delete created files"

build: venv
	# @mkdir -p data

venv:
	@conda create --prefix ./venv python=3.8 -y
	@venv/bin/pip install -r requirements.txt

clean:
	# rm -rf tmp
	rm -rf venv
