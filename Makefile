.PHONY: venv tests
venv:
	python -m venv venv

deps: venv
	. ./venv/bin/activate && pip install -r requirements.txt

clean:
	rm -r venv

gui:
	pyuic5 klena/gui.ui -o klena/gui.py

tests:
	python tests/test_main.py

