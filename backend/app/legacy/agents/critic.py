from __future__ import annotations

import json

from app.agents.common import load_prompt
from app.logging_config import logger
from app.models.response import Critique, ClaimBucket, ResponsePlan, SimulationResult
from app.services.llm_service import LLMError, LLMService
from app.simulation.engine import simulate_plan
from app.services.crisis_service import crisis_service


class CriticAgent:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def critique(self, disaster_type: str, plan: ResponsePlan) -> Critique:
        world = crisis_service.get_world(disaster_type)
        outcome = simulate_plan(world, plan)
        sim: SimulationResult = outcome["simulation"]
        fallback = Critique(
            strengths=[
                f"Modeled score {sim.overall_score}",
                f"People protected {sim.people_evacuated}",
            ],
            weaknesses=sim.bottlenecks or ["No bottleneck list produced; still review shelter headroom."],
            risks=sim.unsafe_routes,
            missing_information=["Field confirmation of road status", "Live hospital divert policy"],
            recommendations=[
                "Re-run simulation after any new event ingestion.",
                "Do not treat score as a guarantee of lives saved.",
            ],
            confidence=0.72,
            claim_buckets=ClaimBucket(
                observed_facts=sim.evidence,
                inferences=["Bottlenecks listed are from the evacuation flow model."],
                recommendations=["Add a backup corridor if unsafe_routes is non-empty."],
                assumptions=["Prototype capacities are synthetic."],
            ),
            llm_available=False,
            mode="deterministic_briefing",
        )
        if not self.llm.available:
            return fallback
        try:
            result = self.llm.generate_structured(
                json.dumps(
                    {"plan": plan.model_dump(mode="json"), "simulation": sim.model_dump(mode="json")},
                    default=str,
                )[:12000],
                Critique,
                system=load_prompt("critic.txt"),
            )
            result.llm_available = True
            result.mode = "llm"
            if not result.weaknesses:
                result.weaknesses = fallback.weaknesses
            return result
        except LLMError:
            logger.error("Critic LLM failed")
            return fallback
