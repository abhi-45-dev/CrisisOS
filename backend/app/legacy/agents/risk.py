from __future__ import annotations

import json

from app.agents.common import load_prompt
from app.logging_config import logger
from app.models.response import AIAnalysis, ClaimBucket
from app.services.llm_service import LLMError, LLMService
from app.tools.crisis_tools import calculate_risk


class RiskAgent:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def explain(self, disaster_type: str, zone_id: str | None = None) -> AIAnalysis:
        risks = calculate_risk(disaster_type, zone_id)
        rows = risks if isinstance(risks, list) else [risks]
        facts = [
            f"{row['zone_name']} score {row['risk_score']} ({row['risk_level']}) factors={row['contributing_factors']}"
            for row in rows[:8]
        ]
        fallback = AIAnalysis(
            summary="Deterministic risk engine results for operator review.",
            observations=facts,
            risks=[f"{row['zone_id']} is {row['risk_level']}" for row in rows if row["risk_level"] in {"high", "critical"}],
            recommendations=["Do not treat these scores as forecasts of casualties; they are relative hazard indices."],
            evidence=[item for row in rows[:5] for item in row.get("evidence", [])],
            confidence=0.8,
            assumptions=["Weights are prototype heuristics, not calibrated loss models."],
            critical_zones=[row["zone_id"] for row in rows if row["risk_level"] == "critical"],
            claim_buckets=ClaimBucket(observed_facts=facts, inferences=[], recommendations=[], assumptions=[]),
            llm_available=self.llm.available,
            mode="deterministic_briefing" if not self.llm.available else "llm",
        )
        if not self.llm.available:
            return fallback
        try:
            result = self.llm.generate_structured(
                f"Explain these backend-calculated risks. Do not change the numbers.\n{json.dumps(rows[:8], default=str)}",
                AIAnalysis,
                system=load_prompt("analyst.txt"),
            )
            result.evidence = fallback.evidence
            result.critical_zones = fallback.critical_zones
            result.llm_available = True
            result.mode = "llm"
            return result
        except LLMError:
            logger.error("Risk agent LLM failed")
            fallback.mode = "deterministic_briefing"
            fallback.llm_available = False
            return fallback
