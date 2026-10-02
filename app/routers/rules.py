import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.dependencies import get_rulebook
from app.schemas import CompiledRuleOut, CompileRequest, RulebookOut
from app.services.documents import resolve_document_for_port
from app.services.rulebook import (
    CompileOutcome,
    RulebookService,
    outcome_from_cache,
    prompt_version,
)

router = APIRouter(prefix="/v1/rules", tags=["rules"])


@router.post("/compile", response_model=RulebookOut)
async def compile_rules(
    request: CompileRequest,
    session: AsyncSession = Depends(get_session),
    rulebook: RulebookService = Depends(get_rulebook),
) -> RulebookOut:
    """Research, extract, validate and review the rules for a port's charges.

    Compiling a port from scratch makes many model calls and can take a few
    minutes; cached rules return at once. scripts/compile_rules.py does the
    same from the command line."""
    document, port_key = await resolve_document_for_port(
        session, request.port, document_id=request.document_id
    )
    report = await rulebook.compile_port(
        document,
        port_key,
        charge_ids=request.charge_ids,
        include_all=request.include_all,
        refresh=request.refresh,
    )
    usage = report.usage
    return RulebookOut(
        document_id=report.document_id,
        port=report.port_key,
        prompt_version=report.prompt_version,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        rules=[_rule_out(outcome) for outcome in report.outcomes],
    )


@router.get("", response_model=RulebookOut)
async def get_rulebook_for_port(
    port: str = Query(min_length=1),
    document_id: uuid.UUID | None = None,
    session: AsyncSession = Depends(get_session),
    rulebook: RulebookService = Depends(get_rulebook),
) -> RulebookOut:
    """The rules compiled so far for a port, for review."""
    document, port_key = await resolve_document_for_port(session, port, document_id=document_id)
    rows = await rulebook.cached_rules(document.id, port_key)
    return RulebookOut(
        document_id=document.id,
        port=port_key,
        prompt_version=prompt_version(),
        prompt_tokens=0,
        completion_tokens=0,
        rules=[_rule_out(outcome_from_cache(row)) for row in rows],
    )


def _rule_out(outcome: CompileOutcome) -> CompiledRuleOut:
    return CompiledRuleOut(
        charge_id=outcome.charge_id,
        name=outcome.name,
        status=outcome.status,
        from_cache=outcome.from_cache,
        revisions=outcome.revisions,
        issues=outcome.issues,
        review_notes=outcome.review_notes,
        sections_read=outcome.sections_read,
        research_notes=outcome.research_notes,
        model=outcome.model,
        compiled_at=outcome.compiled_at,
        error=outcome.error,
        rule=outcome.rule.model_dump(mode="json") if outcome.rule else None,
    )
