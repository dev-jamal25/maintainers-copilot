"""ToolService: typed dispatch for all 5 tools + structured ToolError on bad input / backend."""

from __future__ import annotations

from uuid import UUID, uuid4

from app.domain.exceptions import ExternalServiceUnavailable
from app.domain.memory import EpisodicMemory, MemoryWriteRequest
from app.domain.tools import (
    AnswerWithRagInput,
    AnswerWithRagOutput,
    ClassifyIssueOutput,
    ExtractEntitiesOutput,
    SummarizeThreadOutput,
    ToolError,
    ToolName,
    WriteMemoryOutput,
)
from app.infra.errors import ModelServerUnavailableError
from app.services.tool_service import ToolService, tool_specs


class FakeModelServer:
    def __init__(self, *, down: bool = False) -> None:
        self.down = down

    async def classify(self, title: str, issue_body: str = "") -> tuple[str, float | None]:
        if self.down:
            raise ModelServerUnavailableError("down")
        return "bug", 0.8

    async def extract_entities(self, text: str, *, max_entities: int = 50) -> list[dict[str, str]]:
        return [{"text": "Airflow", "type": "ORG"}]

    async def summarize(self, text: str) -> str:
        return "short summary"


class FakeRag:
    def __init__(self, *, down: bool = False) -> None:
        self.down = down

    async def answer(
        self, payload: AnswerWithRagInput, *, request_id: str | None = None
    ) -> AnswerWithRagOutput:
        if self.down:
            raise ExternalServiceUnavailable("rag down")
        return AnswerWithRagOutput(answer="grounded", citations=[])


class FakeMemory:
    def __init__(self) -> None:
        self.written: MemoryWriteRequest | None = None
        self.user_id: UUID | None = None

    async def write_memory(
        self,
        user_id: UUID,
        request: MemoryWriteRequest,
        *,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> EpisodicMemory:
        from datetime import UTC, datetime

        self.written = request
        self.user_id = user_id
        return EpisodicMemory(
            id=uuid4(),
            user_id=user_id,
            event_type=request.event_type,
            content=request.content,
            subject=request.subject,
            created_at=datetime.now(UTC),
        )


def _service(*, ms_down: bool = False, rag_down: bool = False) -> tuple[ToolService, FakeMemory]:
    memory = FakeMemory()
    service = ToolService(
        model_server=FakeModelServer(down=ms_down),
        rag=FakeRag(down=rag_down),
        memory=memory,
    )
    return service, memory


async def test_classify_tool() -> None:
    service, _ = _service()
    outcome = await service.run(
        ToolName.CLASSIFY_ISSUE, {"title": "crash", "body": "boom"}, user_id=uuid4()
    )
    assert not outcome.is_error
    assert isinstance(outcome.result, ClassifyIssueOutput)
    assert outcome.result.label.value == "bug"


async def test_extract_and_summarize_tools() -> None:
    service, _ = _service()
    entities = await service.run(
        ToolName.EXTRACT_ENTITIES, {"text": "Airflow scheduler"}, user_id=uuid4()
    )
    assert isinstance(entities.result, ExtractEntitiesOutput)
    summary = await service.run(
        ToolName.SUMMARIZE_THREAD, {"text": "a long thread"}, user_id=uuid4()
    )
    assert isinstance(summary.result, SummarizeThreadOutput)


async def test_answer_with_rag_tool() -> None:
    service, _ = _service()
    outcome = await service.run(ToolName.ANSWER_WITH_RAG, {"question": "why?"}, user_id=uuid4())
    assert isinstance(outcome.result, AnswerWithRagOutput)
    assert outcome.result.answer == "grounded"


async def test_write_memory_tool_passes_user_and_payload() -> None:
    service, memory = _service()
    user_id = uuid4()
    outcome = await service.run(
        ToolName.WRITE_MEMORY,
        {"event_type": "preference", "content": "prefers concise replies"},
        user_id=user_id,
    )
    assert isinstance(outcome.result, WriteMemoryOutput)
    assert memory.user_id == user_id
    assert memory.written is not None and memory.written.content == "prefers concise replies"


async def test_invalid_input_returns_tool_error_not_raises() -> None:
    service, _ = _service()
    outcome = await service.run(ToolName.CLASSIFY_ISSUE, {"body": "no title"}, user_id=uuid4())
    assert outcome.is_error
    assert isinstance(outcome.result, ToolError)
    assert outcome.result.code == "invalid_input"
    assert outcome.result.recoverable is False


async def test_model_server_down_returns_recoverable_tool_error() -> None:
    service, _ = _service(ms_down=True)
    outcome = await service.run(ToolName.CLASSIFY_ISSUE, {"title": "crash"}, user_id=uuid4())
    assert outcome.is_error
    assert isinstance(outcome.result, ToolError)
    assert outcome.result.recoverable is True


async def test_rag_down_returns_tool_error() -> None:
    service, _ = _service(rag_down=True)
    outcome = await service.run(ToolName.ANSWER_WITH_RAG, {"question": "q"}, user_id=uuid4())
    assert outcome.is_error
    assert isinstance(outcome.result, ToolError)


def test_tool_specs_cover_all_tools() -> None:
    specs = tool_specs()
    names = {spec["name"] for spec in specs}
    assert names == {t.value for t in ToolName}
    assert all("input_schema" in spec for spec in specs)
