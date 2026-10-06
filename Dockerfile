FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY config ./config
COPY scripts ./scripts
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
ENV PATH="/app/.venv/bin:$PATH"
# The data pipeline runs at first start (downloads the UCI file; about ten minutes and up to ~8 GB of
# memory, set QSTATS_WORKERS lower on a small machine) unless data/ holds a completed run. A failed
# step stops the container; the marker file is written only after the last step succeeds.
CMD ["sh", "-c", "set -e; if [ ! -f data/.pipeline_complete ]; then python scripts/download_data.py; python scripts/prepare_transactions.py; python scripts/build_demo_population.py; python scripts/simulate_supply_chain.py; python scripts/initialize_db.py; python scripts/run_planning_cycle.py; touch data/.pipeline_complete; fi; exec streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
