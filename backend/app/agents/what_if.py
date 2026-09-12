from __future__ import annotations

import json

from app.agents.common import load_prompt
from app.exceptions import InvalidScenarioError, LLMError
from app.logging_config import logger
from app.models.response import ClaimBucket, ScenarioChange, WhatIfExplanation, WhatIfResult
from app.services.llm_service import LLMService
from app.services.scenario_service import scenario_service
from pydantic import BaseModel, Field


class ParsedChanges(BaseModel):
    changes: list[ScenarioChange] = Field(default_factory=list)


class WhatIfAgent:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def run(self, disaster_type: str, question: str) -> WhatIfResult:
        changes = self._parse(question, disaster_type)
        raw = scenario_service.run_scenario(disaster_type, question, changes)
        before_evac = raw["changed_evacuation"]["before"]
        after_evac = raw["changed_evacuation"]["after"]
        impacts = [
            f"Modeled people evacuated {before_evac['people_evacuated']} -> {after_evac['people_evacuated']}",
            f"Evacuation completion {before_evac['evacuation_completion']} -> {after_evac['evacuation_completion']}",
            f"Plan-like overall score {raw['before_sim']['overall_score']} -> {raw['after_sim']['overall_score']}",
        ]
        for row in raw["changed_risk"][:6]:
            impacts.append(
                f"{row['zone_name']} risk {row['before']} -> {row['after']} (delta {row['delta']})"
            )
        summary = (
            f"Scenario '{question}' applied {len(changes)} structured change(s) without mutating live state. "
            + "; ".join(impacts[:3])
        )
        recommendations = [
            "Keep the original crisis state; this run used a cloned world.",
            "If a key corridor is lost, pre-stage inland shelters and alternate routes.",
        ]
        result = WhatIfResult(
            scenario=raw["scenario"],
            before={
                "average_risk": raw["before_sim"]["average_risk"],
                "evacuation": before_evac,
                "overall_score": raw["before_sim"]["overall_score"],
            },
            after={
                "average_risk": raw["after_sim"]["average_risk"],
                "evacuation": after_evac,
                "overall_score": raw["after_sim"]["overall_score"],
            },
            changes=changes,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.73 if not self.llm.available else 0.8,
            affected_zones=raw["affected_zones"],
            changed_routes=raw["changed_routes"],
            changed_risk=raw["changed_risk"],
            changed_resource_allocation=raw["changed_resource_allocation"],
            changed_evacuation=raw["changed_evacuation"],
            impact_summary=summary,
            llm_available=self.llm.available,
            mode="llm" if self.llm.available else "deterministic_briefing",
            claim_buckets=ClaimBucket(
                observed_facts=impacts,
                inferences=["Deltas come from cloned-state simulation, not live mutation."],
                recommendations=recommendations,
                assumptions=["Parser may simplify compound natural-language requests."],
            ),
        )
        if self.llm.available:
            try:
                explained = self.llm.generate_structured(
                    json.dumps(
                        {
                            "question": question,
                            "changes": [c.model_dump(mode="json") for c in changes],
                            "impacts": impacts,
                            "changed_risk": raw["changed_risk"],
                        },
                        default=str,
                    )[:10000],
                    WhatIfExplanation,
                    system=load_prompt("what_if.txt"),
                )
                result.impact_summary = explained.impact_summary or result.impact_summary
                result.recommendations = explained.recommendations or result.recommendations
                result.confidence = explained.confidence
                result.mode = "llm"
            except LLMError:
                logger.error("What-if explanation LLM failed; keeping deterministic summary")
                result.llm_available = False
                result.mode = "deterministic_briefing"
        return result

    def _parse(self, question: str, disaster_type: str) -> list[ScenarioChange]:
        parsed: list[ScenarioChange] = []
        if self.llm.available:
            try:
                structured = self.llm.generate_structured(
                    f"Extract scenario changes for disaster={disaster_type}. User: {question}",
                    ParsedChanges,
                    system=load_prompt("what_if.txt"),
                )
                parsed = structured.changes
            except LLMError:
                logger.error("What-if parse LLM failed; using rules")
        if parsed:
            return parsed
        try:
            return scenario_service.parse_natural_language(question, disaster_type)
        except InvalidScenarioError:
            if parsed:
                return parsed
            raise
