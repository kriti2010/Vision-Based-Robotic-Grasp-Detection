.PHONY: test smoke data train rounds episodes analyze demo clean

PYTHON ?= python

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) main.py --config configs/smoke.yaml --stage smoke

data:
	$(PYTHON) main.py --config configs/default.yaml --stage data

train:
	$(PYTHON) main.py --config configs/default.yaml --stage train

rounds:
	$(PYTHON) main.py --config configs/default.yaml --stage rounds

episodes:
	$(PYTHON) main.py --config configs/default.yaml --stage episodes

analyze:
	$(PYTHON) main.py --config configs/default.yaml --stage analyze

demo:
	streamlit run demo/app.py

ui:
	$(PYTHON) ui/server.py 8080

clean:
	rm -rf data/ checkpoints/ results/ .pytest_cache/
