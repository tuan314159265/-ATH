PYTHON=python3

analysis:
	$(PYTHON) data_characteristic.py

visualize:
	$(PYTHON) visualization.py

install:
	pip install -r requirements.txt


