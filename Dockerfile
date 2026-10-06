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
# The data pipeline runs at first start (downloads the UCI file, ~2 minutes) unless data/ is mounted.
CMD ["sh", "-c", "[ -f data/planner.sqlite ] || (python scripts/download_data.py && python scripts/prepare_transactions.py && python scripts/build_demo_population.py && python scripts/simulate_supply_chain.py && python scripts/initialize_db.py && python scripts/run_planning_cycle.py); streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
