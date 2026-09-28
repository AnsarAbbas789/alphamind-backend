"""
AlphaMind — llm/ensemble.py
=====================================================================
تمام فعال LLMs کو بیک وقت (parallel/asyncio) کال کریں، اور ان کے
ووٹ اکٹھے کریں۔ فیصلہ کن اصول:

  1. کم از کم 2 providers کا جواب آنا لازمی ہے، ورنہ سگنل نہیں بنے گا
     (صرف 1 provider پر بھروسہ کبھی نہیں کیا جائے گا)۔
  2. واضح اکثریت (>50%) درکار ہے۔ برابری (tie) کی صورت میں سگنل
     خودکار NEUTRAL ہو جائے گا — کبھی اندازے سے سمت طے نہیں ہوگی۔
  3. حتمی confidence = (اکثریت کا تناسب) × (AI کی اوسط confidence) ×
     (rule-engine کا bias confidence) — تینوں یکجا، کسی ایک ذریعے پر
     انحصار نہیں۔
  4. ہر ووٹ کی reasoning محفوظ رکھی جاتی ہے (transparency کے لیے) —
     یوزر دیکھ سکتا ہے کس AI نے کیا کہا۔
=====================================================================
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx

from app.llm.base import BaseLLMProvider, LLMVote

MIN_PROVIDERS_REQUIRED = 2


@dataclass
class EnsembleResult:
    final_signal: str = "NEUTRAL"
    agreement_ratio: float = 0.0
    votes: list[LLMVote] = field(default_factory=list)
    responded_count: int = 0
    total_providers: int = 0
    avg_ai_confidence: float = 0.0
    quorum_met: bool = False
    tie: bool = False


async def run_ensemble(providers: list[BaseLLMProvider], prompt: str) -> EnsembleResult:
    result = EnsembleResult(total_providers=len(providers))

    if not providers:
        return result

    async with httpx.AsyncClient() as client:
        votes = await asyncio.gather(*[p.get_vote(client, prompt) for p in providers])

    result.votes = list(votes)
    valid_votes = [v for v in votes if v.ok]
    result.responded_count = len(valid_votes)

    if result.responded_count < MIN_PROVIDERS_REQUIRED:
        result.final_signal = "NEUTRAL"
        result.quorum_met = False
        return result

    result.quorum_met = True

    long_votes = [v for v in valid_votes if v.signal == "LONG"]
    short_votes = [v for v in valid_votes if v.signal == "SHORT"]
    neutral_votes = [v for v in valid_votes if v.signal == "NEUTRAL"]

    counts = {"LONG": len(long_votes), "SHORT": len(short_votes), "NEUTRAL": len(neutral_votes)}
    winner = max(counts, key=counts.get)
    winner_count = counts[winner]

    others_max = max(c for k, c in counts.items() if k != winner)
    result.tie = winner_count == others_max

    if result.tie:
        result.final_signal = "NEUTRAL"
    else:
        result.final_signal = winner

    result.agreement_ratio = round(winner_count / result.responded_count, 3)

    winning_votes = {"LONG": long_votes, "SHORT": short_votes, "NEUTRAL": neutral_votes}[winner]
    if winning_votes:
        result.avg_ai_confidence = round(
            sum(v.confidence for v in winning_votes if v.confidence is not None) / len(winning_votes), 1
        )

    return result