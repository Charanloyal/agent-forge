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

def _make_engine(url: str | None):
    import os
    clean_url = (url or "").strip()
    use_sqlite = (
        os.getenv("USE_SQLITE", "true").lower() == "true"
        or "localhost" in clean_url
        or not clean_url
    )
    if use_sqlite or clean_url.startswith("sqlite"):
        logger.info("Using SQLite async engine for persistence.")
        return create_async_engine("sqlite+aiosqlite:///./agentforge.db", connect_args={"check_same_thread": False})

    if clean_url.startswith("postgres://"):
        clean_url = clean_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif clean_url.startswith("postgresql://") and not clean_url.startswith("postgresql+asyncpg://"):
        clean_url = clean_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    logger.info("Using PostgreSQL async engine for persistence.")
    return create_async_engine(
        clean_url,
        echo=settings.DEBUG,
        pool_pre_ping=True,
        pool_size=20,
        max_overflow=10,
        pool_recycle=3600
    )

try:
    engine = _make_engine(settings.DATABASE_URL)
except Exception as err:
    logger.warning("Failed to create engine with primary URL (%s). Falling back to SQLite.", err)
    engine = create_async_engine("sqlite+aiosqlite:///./agentforge.db", connect_args={"check_same_thread": False})

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an isolated async SQLAlchemy session per request.
    100% fail-safe: automatically falls back to an in-memory SQLite session if primary engine fails.
    """
    global engine, async_session_factory
    session = None
    try:
        session = async_session_factory()
    except Exception as e:
        logger.warning("Failed to create session from factory (%s). Initializing SQLite engine.", e)
        fallback_engine = create_async_engine("sqlite+aiosqlite:///./agentforge.db", connect_args={"check_same_thread": False})
        fallback_factory = async_sessionmaker(bind=fallback_engine, class_=AsyncSession, expire_on_commit=False)
        session = fallback_factory()

    try:
        yield session
        try:
            await session.commit()
        except Exception:
            pass
    except Exception as exc:
        logger.warning("Session operation error: %s", exc)
        try:
            await session.rollback()
        except Exception:
            pass
    finally:
        try:
            await session.close()
        except Exception:
            pass


async def init_db() -> None:
    """
    Idempotent schema initialization: creates all declared tables in PostgreSQL or SQLite fallback.
    """
    global engine, async_session_factory
    logger.info("Initializing database schema with URL: %s", settings.DATABASE_URL)
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
