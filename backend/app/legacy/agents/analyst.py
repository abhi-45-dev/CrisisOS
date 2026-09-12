from __future__ import annotations

import json

from app.agents.common import load_prompt
from app.logging_config import logger
from app.models.response import AIAnalysis, ClaimBucket
from app.services.llm_service import LLMError, LLMService
from app.tools.crisis_tools import briefing_context


class AnalystAgent:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def analyze(self, disaster_type: str, question: str | None = None) -> AIAnalysis:
        context = briefing_context(disaster_type)
        fallback = self._from_tools(disaster_type, context, question)
        if not self.llm.available:
            logger.info("Analyst using deterministic briefing; LLM unavailable")
            return fallback
        prompt = (
            f"Question: {question or 'What are the most critical areas right now?'}\n\n"
            f"BACKEND TOOL OUTPUT (facts only):\n{json.dumps(context, default=str)[:12000]}"
        )
        try:
            result = self.llm.generate_structured(prompt, AIAnalysis, system=load_prompt("analyst.txt"))
            result.llm_available = True
            result.mode = "llm"
            result.evidence = list(dict.fromkeys(result.evidence + fallback.evidence))
            result.critical_zones = result.critical_zones or fallback.critical_zones
            return result
        except LLMError:
            logger.error("Analyst LLM failed; returning deterministic briefing")
            fallback.assumptions.append("Narrative produced from deterministic tools after LLM failure.")
            return fallback

    def _from_tools(self, disaster_type: str, context: dict, question: str | None) -> AIAnalysis:
        top = context["top_risks"]
        critical = [row["zone_id"] for row in top if row["risk_level"] == "critical"]
        observations = [
            f"{row['zone_name']} risk {row['risk_score']} ({row['risk_level']})"
            for row in top[:5]
        ]
        dash = context["dashboard"]
        facts = [
            f"Disaster {disaster_type} severity {dash['overall_severity']}",
            f"Affected population {dash['affected_population']}",
            f"Active incidents {dash['active_incidents']}",
            f"Blocked roads {dash['blocked_roads']}",
        ]
        return AIAnalysis(
            summary=(
                f"{disaster_type.title()} crisis is {dash['status']} with severity {dash['overall_severity']}. "
                f"Critical zones: {', '.join(critical) or 'none'}. "
                f"{dash['affected_population']} people sit in currently affected zones."
            ),
            observations=observations,
            risks=[f"{z} is critical" for z in critical] + [f"High-risk zones: {dash['high_risk_zones']}"],
            recommendations=[
                "Prioritize evacuation and resources for critical zones using backend routing, not ad-hoc paths.",
                "Keep a human operator in the loop before any field deployment.",
            ],
            evidence=facts + [str(row.get("evidence")) for row in top[:3]],
            confidence=0.74,
            assumptions=["Risk scores come from the deterministic risk engine, not an LLM."],
            critical_zones=critical,
            uncertainties=["Incident reports with confidence below 0.7 may be incomplete."],
            claim_buckets=ClaimBucket(
                observed_facts=facts,
                inferences=[
                    "Zones with both high population and high hazard parameters will dominate casualties if unaddressed."
                ],
                recommendations=["Focus operator attention on the listed critical zones first."],
                assumptions=["Synthetic Harbor City data is internally consistent but not real telemetry."],
            ),
            llm_available=False,
            mode="deterministic_briefing",
        )

from app.services.llm_service import llm_service
analyst_agent = AnalystAgent(llm_service)
