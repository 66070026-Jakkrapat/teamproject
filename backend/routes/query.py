"""Query API route — main agent interface."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.services.agent import agent_query

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


@router.post("")
async def query_agent(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit a question to the AI Agent.

    The agent will classify the query and route it to either:
    - Vector Search (semantic questions)
    - Text-to-SQL (analytical questions)
    - Hybrid (both)
    """
    result = await agent_query(request.question, db)
    return result
