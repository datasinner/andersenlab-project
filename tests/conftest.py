import os

# The whole app stack (Settings, the SQLAlchemy engine, Alembic's env.py) is
# constructed at import time, so the test database and the fake LLM
# provider must be selected before anything under app.* is imported -
# hence these run as the very first thing pytest loads, and every app.*
# import in fixtures is deferred into fixture bodies.
os.environ["DATABASE_URL"] = "postgresql+asyncpg://tariff:tariff@localhost:5432/test"
os.environ["LLM_PROVIDER"] = "fake"
os.environ["APP_ENV"] = "test"
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["API_AUTH_KEY"] = ""
