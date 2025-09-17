PYTHON=python3

analysis:
	$(PYTHON) data_characteristic.py

generate:
	$(PYTHON) data_generator.py

visualize:
	$(PYTHON) visualization.py

install:
	pip install -r requirements.txt

clean:
	rm -f dataset/*.csv
