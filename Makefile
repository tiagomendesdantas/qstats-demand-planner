# QStats Demand & Inventory Planner
#   make demo      acquire data -> clean -> sample -> simulate -> database -> plan -> open the app
#   make test      run the test suite
PY := uv run python

.PHONY: install data population simulate db plan app api demo test lint requirements clean

install:
	uv sync

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/prepare_transactions.py

population:
	$(PY) scripts/build_demo_population.py

simulate:
	$(PY) scripts/simulate_supply_chain.py

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
