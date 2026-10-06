# QStats Demand & Inventory Planner
#   make demo      acquire data -> clean -> sample -> simulate -> database -> plan -> open the app
#   make test      run the test suite
#   make demo WORKERS=4   fewer parallel simulated worlds (each takes about 1 GB of memory)
PY := uv run python
WORKERS ?=

.PHONY: install data population simulate db plan app api demo test lint requirements clean
# the pipeline steps depend on each other in the order listed under `demo`
.NOTPARALLEL:

install:
	uv sync

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/prepare_transactions.py

population:
	$(PY) scripts/build_demo_population.py

simulate:
	$(PY) scripts/simulate_supply_chain.py $(if $(WORKERS),--workers $(WORKERS))

db:
	$(PY) scripts/initialize_db.py

plan:
	$(PY) scripts/run_planning_cycle.py

app:
	uv run streamlit run dashboard/streamlit_app.py

api:
	uv run uvicorn qstats_planner.api.main:app --reload --port 8000

demo: install data population simulate db plan app

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests dashboard --statistics

requirements:
	uv export --no-hashes --no-dev --format requirements-txt > requirements.txt

clean:
	rm -rf data/processed data/simulation data/planner.sqlite
