from app.agents.analyst import AnalystAgent
from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.resource import ResourceAgent
from app.agents.risk import RiskAgent
from app.agents.what_if import WhatIfAgent
from app.services.llm_service import llm_service

analyst_agent = AnalystAgent(llm_service)
risk_agent = RiskAgent(llm_service)
resource_agent = ResourceAgent(llm_service)
planner_agent = PlannerAgent(llm_service)
critic_agent = CriticAgent(llm_service)
what_if_agent = WhatIfAgent(llm_service)

__all__ = [
    "analyst_agent",
    "critic_agent",
    "planner_agent",
    "resource_agent",
    "risk_agent",
    "what_if_agent",
]
