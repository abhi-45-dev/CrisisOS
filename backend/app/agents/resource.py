from __future__ import annotations

import json

from app.agents.common import load_prompt
from app.logging_config import logger
from app.models.response import AIAnalysis, ClaimBucket
from app.services.llm_service import LLMError, LLMService
from app.simulation.resources import allocate_resources
from app.tools.crisis_tools import get_available_resources
from app.services.crisis_service import crisis_service
from app.simulation.risk import apply_risk_to_world


class ResourceAgent:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def analyze(self, disaster_type: str) -> AIAnalysis:
        available = get_available_resources(disaster_type)
        world = crisis_service.clone_world(disaster_type)
        apply_risk_to_world(world)
        allocation = allocate_resources(world)
        facts = [
            f"{item['id']} {item['type']} available {item['available_quantity']}/{item['quantity']} in {item['zone_id']}"
            for item in available
        ]
        shortages = allocation.unmet_demand.get("resource_shortages", {})
        fallback = AIAnalysis(
            summary=f"Resource picture for {disaster_type}: {len(available)} stocks with remaining units; allocation score {allocation.allocation_score}.",
            observations=facts[:12],
            risks=[f"Unmet demand in {zone}: {need}" for zone, need in list(shortages.items())[:8]],
            recommendations=[
                "Fill unmet demand in highest-risk zones first.",
                "Do not assign more units than available_quantity.",
            ],
            evidence=allocation.evidence,
            confidence=0.77,
            assumptions=["Travel times use the road graph; air assets ignore blocked roads."],
            claim_buckets=ClaimBucket(
                observed_facts=facts[:10],
                inferences=["Shortages listed are from the greedy allocator, not an LLM."],
                recommendations=["Stage unused caches from low-risk zones toward critical zones."],
                assumptions=[],
            ),
            llm_available=False,
            mode="deterministic_briefing",
        )
        if not self.llm.available:
            return fallback
        try:
            result = self.llm.generate_structured(
                json.dumps({"available": available, "allocation": allocation.model_dump(mode="json")}, default=str)[:12000],
                AIAnalysis,
                system=load_prompt("analyst.txt"),
            )
            result.evidence = allocation.evidence
            result.llm_available = True
            result.mode = "llm"
            return result
        except LLMError:
            logger.error("Resource agent LLM failed")
            return fallback
