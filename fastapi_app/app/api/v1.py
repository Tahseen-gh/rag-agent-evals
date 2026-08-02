from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fastapi_app.app.config import settings
from rag.agent import get_agent_llm_client, run_agent
from rag.generation import answer_question, get_llm_client

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


class RagAnswerRequest(BaseModel):
    question: str
    backend: str = "hybrid"
    k: int = 4


class CitationOut(BaseModel):
    number: int
    doc_id: str
    snippet: str


class RagAnswerResponse(BaseModel):
    question: str
    answer: str
    backend: str
    citations: list[CitationOut]


@router.post("/rag/answer", response_model=RagAnswerResponse)
def rag_answer(req: RagAnswerRequest) -> RagAnswerResponse:
    client = get_llm_client(
        use_dummy=settings.use_dummy_llm,
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
    )
    try:
        result = answer_question(
            req.question,
            backend_name=req.backend,
            k=req.k,
            llm_client=client,
            embedding_model=settings.embedding_model,
            hybrid_alpha=settings.hybrid_alpha,
            use_dummy_embeddings=settings.use_dummy_embeddings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid backend") from exc

    return RagAnswerResponse(
        question=result.question,
        answer=result.answer,
        backend=result.backend,
        citations=[
            CitationOut(number=c.number, doc_id=c.doc_id, snippet=c.snippet)
            for c in result.citations
        ],
    )


class AgentAskRequest(BaseModel):
    question: str
    backend: str = "hybrid"
    max_turns: int | None = None


class AgentStepOut(BaseModel):
    kind: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    text: str | None = None


class AgentAskResponse(BaseModel):
    question: str
    answer: str
    steps: list[AgentStepOut]
    turns_used: int
    stopped_reason: str


@router.post("/agent/ask", response_model=AgentAskResponse)
def agent_ask(req: AgentAskRequest) -> AgentAskResponse:
    client = get_agent_llm_client(
        use_dummy=settings.use_dummy_llm,
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
    )
    try:
        result = run_agent(
            req.question,
            llm_client=client,
            backend_name=req.backend,
            embedding_model=settings.embedding_model,
            hybrid_alpha=settings.hybrid_alpha,
            use_dummy_embeddings=settings.use_dummy_embeddings,
            max_turns=req.max_turns or settings.agent_max_turns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid backend") from exc

    return AgentAskResponse(
        question=result.question,
        answer=result.answer,
        steps=[
            AgentStepOut(
                kind=s.kind,
                tool_name=s.tool_name,
                tool_input=s.tool_input,
                tool_output=s.tool_output,
                text=s.text,
            )
            for s in result.steps
        ],
        turns_used=result.turns_used,
        stopped_reason=result.stopped_reason,
    )
