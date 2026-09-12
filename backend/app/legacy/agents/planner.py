from __future__ import annotations

import json
from uuid import uuid4

from app.agents.common import load_prompt
from app.logging_config import logger
from app.models.location import RiskLevel, RoadStatus
from app.models.response import PlanAction, ResponsePlan
from app.services.crisis_service import crisis_service
from app.services.llm_service import LLMError, LLMService
from app.simulation.engine import compare_plans, simulate_plan
from app.simulation.risk import apply_risk_to_world
from app.tools.crisis_tools import briefing_context


class PlannerAgent:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def plan(self, disaster_type: str, objective: str | None = None) -> dict:
        world = crisis_service.get_world(disaster_type)
        apply_risk_to_world(world)
        heuristics = self._heuristic_plans(disaster_type, world)
        llm_plan = None
        if self.llm.available:
            context = briefing_context(disaster_type)
            prompt = (
                f"Objective: {objective or 'Create an evacuation and resource response plan.'}\n"
                f"TOOL FACTS:\n{json.dumps(context, default=str)[:11000]}"
            )
            try:
                llm_plan = self.llm.generate_structured(prompt, ResponsePlan, system=load_prompt("planner.txt"))
                llm_plan.id = llm_plan.id or f"plan_{uuid4().hex[:8]}"
            except LLMError:
                logger.error("Planner LLM failed; using heuristic plans only")
        plans = heuristics if llm_plan is None else [llm_plan, *heuristics]
        for plan in plans:
            if not plan.id:
                plan.id = f"plan_{uuid4().hex[:8]}"
        comparison = compare_plans(world, plans)
        simulated = [simulate_plan(world, plan) for plan in plans]
        return {
            "plans": [item["plan"].model_dump(mode="json") for item in simulated],
            "comparison": comparison,
            "primary": simulated[0]["plan"].model_dump(mode="json"),
            "simulation": simulated[0]["simulation"].model_dump(mode="json"),
            "llm_available": self.llm.available and llm_plan is not None,
        }

    def _heuristic_plans(self, disaster_type: str, world) -> list[ResponsePlan]:
        critical = [z for z in world.zones.values() if z.risk_level == RiskLevel.CRITICAL]
        high = [z for z in world.zones.values() if z.risk_level == RiskLevel.HIGH]
        focus = critical or high or sorted(world.zones.values(), key=lambda z: z.risk_score, reverse=True)[:3]
        blocked = [r for r in world.roads.values() if r.status == RoadStatus.DANGEROUS]
        pumps = disaster_type == "flood"
        actions_a = [PlanAction(type="evacuate_zone", target_id=z.id, description=f"Evacuate {z.name}") for z in focus]
        actions_a.append(PlanAction(type="open_shelter", target_id="s02", description="Keep main stadium shelter open"))
        actions_a.append(PlanAction(type="dispatch_rescue_team", target_id=focus[0].id, quantity=2))
        if pumps:
            actions_a.append(PlanAction(type="deploy_water_pumps", target_id=focus[0].id, quantity=2))
        actions_a.append(PlanAction(type="dispatch_ambulance", target_id=focus[0].id, quantity=2))
        plan_a = ResponsePlan(
            id="plan_evacuate_primary",
            title="Shelter-first evacuation",
            objective="Move people from the highest-risk zones toward North Hills and Greenbelt shelters.",
            actions=actions_a,
            affected_zones=[z.id for z in focus],
            assigned_resources=["res_04", "res_01", "res_08"],
            evidence=["Generated from current risk ranking and shelter locations."],
            confidence=0.7,
            assumptions=["Road graph remains as currently reported unless an action closes a road."],
        )
        actions_b = [PlanAction(type="evacuate_zone", target_id=z.id) for z in focus[:2]]
        for road in blocked[:2]:
            actions_b.append(PlanAction(type="close_road", target_id=road.id, description=f"Close dangerous {road.name}"))
        actions_b.append(PlanAction(type="reroute_evacuation_traffic", target_id="road_22"))
        actions_b.append(PlanAction(type="move_resources", target_id=focus[0].id, quantity=2))
        actions_b.append(PlanAction(type="prioritize_hospitals", target_id="h03"))
        plan_b = ResponsePlan(
            id="plan_corridor_protect",
            title="Protect inland corridors",
            objective="Close dangerous coastal/industrial links and push traffic toward hills hospitals.",
            actions=actions_b,
            affected_zones=[z.id for z in focus],
            assigned_resources=["res_11", "res_05"],
            evidence=["Uses current dangerous-road list from the live graph."],
            confidence=0.66,
            assumptions=["Closing dangerous roads reduces exposure even if travel times increase."],
        )
        return [plan_a, plan_b]
