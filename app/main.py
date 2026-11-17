"""FastAPI entrypoint: mounts the Problem 2 and Problem 3 routers."""
from fastapi import FastAPI

from app.core.logging import setup_logging
from app.model_fallback_system.router import router as fallback_router
from app.lead_extraction_system.router import router as leads_router

setup_logging()

app = FastAPI(title="AI Assignments", version="1.0")
app.include_router(fallback_router)
app.include_router(leads_router)


@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}
