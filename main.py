"""
FastAPI Application — Loan Agentic AI
──────────────────────────────────────
Main entry point. Sets up routes, middleware, and lifecycle events.

Phase 1 Endpoints:
  Auth:
    POST /auth/register          — Create new user account
    POST /auth/login             — Get JWT tokens
    POST /auth/refresh           — Refresh access token
    GET  /auth/me                — Get current user info

  Applications:
    POST /applications           — Submit a loan application
    GET  /applications           — List user's applications
    GET  /applications/{id}      — Get application details

  Documents:
    POST /applications/{id}/upload — Upload PDF document

  Admin:
    GET  /admin/applications     — List all applications (admin only)
    GET  /admin/escalations      — List escalated cases (admin only)

  Health:
    GET  /                       — Health check
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings
from gateway.rate_limiter import limiter
from models.database import create_tables

# Import route modules
from routes import auth_routes, application_routes, admin_routes, process_routes


# ── Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: create tables (dev only — use Alembic in production)
    await create_tables()
    print("✅ Database tables created")
    yield
    # Shutdown: cleanup if needed
    print("👋 Shutting down")


# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Loan Agentic AI",
    description=(
        "An autonomous multi-agent AI system for loan approval. "
        "6 specialised agents (Data, Risk, Policy, Fairness, Explainability, Audit) "
        "collaborate under a LangGraph orchestrator to process applications end-to-end."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.APP_ENV == "development" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Routes ───────────────────────────────────────────────────────────────

app.include_router(auth_routes.router)
app.include_router(application_routes.router)
app.include_router(admin_routes.router)
app.include_router(process_routes.router)


@app.get("/", tags=["Health"])
async def health_check():
    """Service health check."""
    return {
        "service": "Loan Agentic AI",
        "version": "2.0.0",
        "status": "running",
        "environment": settings.APP_ENV,
        "docs": "/docs",
    }
