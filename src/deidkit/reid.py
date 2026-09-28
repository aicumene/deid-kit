# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 Alexandra Bernadotte
"""Пробник реидентификации — ворота перед любым шарингом обезличенного фрагмента.

«Обезличено» — измеренное утверждение, а не свойство алгоритма (2026-08: регекс, вычищавший
имена, модель обошла по году и точной ссылке). Два плеча, обе проверки — на доверенном контуре:

  * ПОИСК (детерминированное плечо): фрагмент как запрос к кускам документов ВСЕХ областей
    (:class:`PassageIndex` — косинусное расстояние по векторам кусков). Считается ранг исходной
    области среди областей выдачи и ЗАПАС — на сколько ближайший кусок своей области ближе
    ближайшего куска чужой. Исходный кусок сидит в индексе, поэтому сырой текст «узнаётся»
    тривиально (запас ≈ его расстояние до чужих) — это контроль; вопрос к обезличенному
    фрагменту — сжался ли запас до нуля, то есть стал ли он неотличим от области той же темы.
    Порог запаса — не «истина», а линейка: на сырых текстах он большой, на честном обезличивании
    стремится к нулю, и отчёт показывает всё распределение.

  * СУДЬЯ (семантическое плечо): ДРУГАЯ модель, чем та, что размечала (иначе самопроверка),
    получает фрагмент и список областей с тем, что о них известно (стороны, адрес, даты,
    суммы — как в картотеке), и называет область или UNKNOWN. Точность против случайного
    угадывания. Судья видит факты — это доверенный контур, факты никуда не уходят.

Модель — любой объект с ``embed``/``generate`` (:class:`deidkit.model.ModelRouter`); индекс
кусков — :class:`PassageIndex` (в памяти — :class:`InMemoryPassageIndex`).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from deidkit.classification import Sensitivity
from deidkit.jsonutil import extract_json
from deidkit.model import QUERY, ModelMessage, ModelRequest, ModelRouter
from deidkit.store import ScopeId

#: Запас косинусного расстояния, начиная с которого «своё ближе чужого» считается узнаванием.
#: Линейка, не истина: см. докстроку модуля.
DEFAULT_MARGIN = 0.05


class PassageIndex(Protocol):
    """Куски документов всех областей с их векторами — то, по чему ищет плечо поиска.

    Расстояние — косинусное (``1 − cos``), как у ``cosine_distance`` pgvector; куски без вектора
    в выдачу не попадают; ``exclude_chunk_id`` (если задан) из выдачи исключается."""

    async def nearest(self, vector: Sequence[float], *, k: int,
                      exclude_chunk_id: Hashable | None = None) -> list[tuple[ScopeId, float]]:
        """``(область, расстояние)`` ``k`` ближайших кусков, по возрастанию расстояния."""
        ...

    async def nearest_in_scope(self, vector: Sequence[float], scope_id: ScopeId, *,
                               exclude_chunk_id: Hashable | None = None) -> float | None:
        """Расстояние до ближайшего куска области ``scope_id``; ``None`` — кусков нет."""
        ...


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """``1 − cos(a, b)``; NaN, если у одного из векторов нулевая длина (как у pgvector)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return math.nan
    return 1.0 - dot / (na * nb)


def _order(d: float) -> tuple[bool, float]:
    # NaN — последним, как в Postgres (NaN больше любого числа).
    return (math.isnan(d), d)


class InMemoryPassageIndex:
    """:class:`PassageIndex` в памяти процесса."""

    def __init__(self) -> None:
        self._rows: list[tuple[Hashable, ScopeId, list[float] | None]] = []

    def add(self, chunk_id: Hashable, scope_id: ScopeId, vector: Sequence[float] | None) -> None:
        self._rows.append((chunk_id, scope_id, list(vector) if vector is not None else None))

    def _candidates(self, exclude_chunk_id: Hashable | None):
        for chunk_id, scope_id, vec in self._rows:
            if vec is None:
                continue
            if exclude_chunk_id is not None and chunk_id == exclude_chunk_id:
                continue
            yield scope_id, vec

    async def nearest(self, vector: Sequence[float], *, k: int,
                      exclude_chunk_id: Hashable | None = None) -> list[tuple[ScopeId, float]]:
        scored = [(scope_id, cosine_distance(vector, vec))
                  for scope_id, vec in self._candidates(exclude_chunk_id)]
        scored.sort(key=lambda sd: _order(sd[1]))
        return scored[:k]

    async def nearest_in_scope(self, vector: Sequence[float], scope_id: ScopeId, *,
                               exclude_chunk_id: Hashable | None = None) -> float | None:
        ds = [cosine_distance(vector, vec)
              for s, vec in self._candidates(exclude_chunk_id) if s == scope_id]
        return min(ds, key=_order) if ds else None


@dataclass
class RetrievalProbe:
    #: Ранг исходной области среди ОБЛАСТЕЙ (не кусков) в top-k; None — в top-k её нет.
    source_rank: int | None
    #: dist(лучший чужой кусок) − dist(лучший свой кусок). > 0 — своё ближе.
    margin: float
    #: Лучшее расстояние по областям, в порядке выдачи.
    top: list[tuple[str, float]] = field(default_factory=list)
    threshold: float = DEFAULT_MARGIN

    @property
    def identifies(self) -> bool:
        return self.source_rank == 1 and self.margin >= self.threshold


async def probe_retrieval(index: PassageIndex, router: ModelRouter, text: str, *,
                          source_scope_id: ScopeId, k: int = 20,
                          threshold: float = DEFAULT_MARGIN,
                          exclude_chunk_id: Hashable | None = None) -> RetrievalProbe:
    """Плечо поиска: ранг и запас исходной области для ``text`` как запроса ко всем областям.

    ``exclude_chunk_id`` — кусок, из которого фрагмент сделан: без исключения фрагмент почти
    дословно повторяет свой кусок и «узнаётся» по тексту, а не по фактам (запас 0.3 и у
    честно обезличенного). С исключением измеряется то, что нужно: тянут ли оставшиеся факты
    к ДРУГИМ документам той же области (адрес, даты, суммы повторяются по всей папке) сильнее,
    чем к области той же темы."""
    vec = (await router.embed([text], sensitivity=Sensitivity.CONFIDENTIAL, kind=QUERY))[0]
    rows = await index.nearest(vec, k=k, exclude_chunk_id=exclude_chunk_id)
    best: dict[ScopeId, float] = {}
    for scope_id, d in rows:
        best.setdefault(scope_id, float(d))
    ordered = sorted(best.items(), key=lambda kv: kv[1])
    ranks = {m: i + 1 for i, (m, _) in enumerate(ordered)}
    own = best.get(source_scope_id)
    if own is None:
        own_row = await index.nearest_in_scope(vec, source_scope_id,
                                               exclude_chunk_id=exclude_chunk_id)
        own = float(own_row) if own_row is not None else 2.0
    others = [d for m, d in ordered if m != source_scope_id]
    margin = (others[0] if others else 2.0) - own
    return RetrievalProbe(source_rank=ranks.get(source_scope_id), margin=margin,
                          top=[(str(m), d) for m, d in ordered], threshold=threshold)


@dataclass(frozen=True)
class Candidate:
    reference: str
    summary: str


@dataclass(frozen=True)
class JudgeWording:
    """How the judge is briefed: who it is and what the candidates are. A deployment sets its own
    once, with :func:`set_judge_wording`; the defaults are neutral."""

    system: str = (
        "You are an analyst who knows the organisation's records well. A colleague shows you an "
        "excerpt. Decide which of the records it comes from, using ONLY facts in the excerpt "
        "(parties, places, dates, amounts, numbers, unique circumstances). The excerpt may be "
        "de-identified; if nothing in it singles out one record, answer UNKNOWN — do not guess from "
        "the subject alone when several records share it.\n"
        "Respond with ONLY a JSON object: {\"reference\": one of the references or \"UNKNOWN\", "
        "\"clue\": the fact that decided it (empty if UNKNOWN)}."
    )
    #: the heading of the candidate list in the judge's prompt
    candidates: str = "Records"


_JUDGE_WORDING = JudgeWording()


def set_judge_wording(wording: JudgeWording) -> None:
    """Set how the re-identification judge is briefed (process-wide)."""
    global _JUDGE_WORDING
    _JUDGE_WORDING = wording


def judge_schema(candidates: list[Candidate]) -> dict:
    refs = [c.reference for c in candidates] + ["UNKNOWN"]
    return {"type": "object",
            "properties": {"reference": {"type": "string", "enum": refs}, "clue": {"type": "string"}},
            "required": ["reference", "clue"], "additionalProperties": False}


@dataclass(frozen=True)
class JudgeVerdict:
    reference: str
    clue: str


async def probe_judge(router: ModelRouter, text: str, candidates: list[Candidate], *,
                      model: str | None = None) -> JudgeVerdict:
    """Плечо судьи: другая модель называет область по фрагменту и картотеке."""
    card = "\n".join(f"- {c.reference}: {c.summary}" for c in candidates)
    request = ModelRequest(
        messages=[
            ModelMessage(role="system", content=_JUDGE_WORDING.system),
            ModelMessage(role="user", content=f"{_JUDGE_WORDING.candidates}:\n{card}\n\nExcerpt:\n{text[:6000]}"),
        ],
        model=model,
        sensitivity=Sensitivity.CONFIDENTIAL,
        temperature=0.0,
        max_tokens=200,
        think=False,
        response_format=judge_schema(candidates),
    )
    response = await router.generate(request)
    data = extract_json(response.text)
    ref = str(data.get("reference") or "UNKNOWN") if isinstance(data, dict) else "UNKNOWN"
    clue = str(data.get("clue") or "") if isinstance(data, dict) else ""
    if ref not in {c.reference for c in candidates}:
        ref = "UNKNOWN"
    return JudgeVerdict(reference=ref, clue=clue[:200])


__all__ = ["Candidate", "DEFAULT_MARGIN", "InMemoryPassageIndex", "JudgeVerdict", "PassageIndex",
           "RetrievalProbe", "cosine_distance", "judge_schema", "probe_judge", "probe_retrieval"]
