import logging
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession
)
from app.config import get_settings
from app.models import Base

logger = logging.getLogger("agentforge.database")
settings = get_settings()

db_url = settings.DATABASE_URL or "sqlite+aiosqlite:///./agentforge.db"
engine_kwargs: dict = {
    "echo": settings.DEBUG,
}

if db_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update({
        "pool_pre_ping": True,
        "pool_size": 20,
        "max_overflow": 10,
        "pool_recycle": 3600
    })

engine = create_async_engine(db_url, **engine_kwargs)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an isolated async SQLAlchemy session per request.
    Rolls back automatically on exception and closes gracefully.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Idempotent schema initialization: creates all declared tables in PostgreSQL or SQLite fallback.
    """
    global engine, async_session_factory
    logger.info("Initializing database schema with URL: %s", db_url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema initialized successfully.")
    except Exception as exc:
        logger.warning("Database initialization failed (%s). Falling back to SQLite.", exc)
        fallback_url = "sqlite+aiosqlite:///./agentforge.db"
        engine = create_async_engine(fallback_url, connect_args={"check_same_thread": False})
        async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("SQLite fallback database schema initialized successfully.")


async def close_db() -> None:
    """Disposes engine connection pool gracefully."""
    logger.info("Closing database engine connection pool...")
    await engine.dispose()
    logger.info("Database connection pool closed.")
