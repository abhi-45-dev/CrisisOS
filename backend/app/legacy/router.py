from uuid import uuid4

from fastapi import APIRouter, Query

from app.agents import analyst_agent, critic_agent, planner_agent, what_if_agent
from app.api.schemas import (
    AnalyzeRequest,
    FloodPredictRequest,
    CompareRequest,
    EventRequest,
    GeoQuery,
    PlanRequest,
    RouteQuery,
    ScenarioCreateRequest,
    SimulateRequest,
    WhatIfRequest,
)
from app.config import settings
from app.models.response import ResponsePlan
from app.services.alert_service import generate_alerts
from app.services.crisis_service import crisis_service, parse_disaster
from app.services.llm_service import llm_service
from app.services.scenario_service import scenario_service
from app.simulation.engine import compare_plans, simulate_plan
from app.simulation.risk import apply_risk_to_world, calculate_zone_risk
from app.simulation.routes import find_safe_routes
from app.tools import crisis_tools
from ml.predictor import model_status, predict_flood
from app.services.weather_service import get_live_rainfall

router = APIRouter()


@router.get("/", tags=["meta"])
def root():
    return {
        "name": settings.APP_NAME,
        "status": "online",
        "version": settings.APP_VERSION,
        "supported_disasters": ["flood", "cyclone", "earthquake"],
        "docs": "/docs",
        "human_in_the_loop": True,
    }


@router.get("/health", tags=["meta"])
def health():
    return {
        "status": "healthy",
        "llm_configured": llm_service.available,
        "crises_loaded": [c.id for c in crisis_service.list_crises()],
    }


@router.get("/ml/status", tags=["ml"])
def ml_status():
    result = model_status()
    weather = get_live_rainfall()
    result["live_weather"] = {
        "source": weather.get("source"),
        "available": weather.get("available"),
        "rainfall_24h_mm": weather.get("rainfall_24h_mm"),
        "observed_at": weather.get("observed_at"),
        "latitude": weather.get("latitude"),
        "longitude": weather.get("longitude"),
    }
    if result.get("trained") and weather.get("available"):
        try:
            current = predict_flood(
                state="TAMIL NADU",
                rainfall_mm=float(weather.get("rainfall_24h_mm") or 0.0),
                month=__import__("datetime").datetime.now().month,
                latitude=float(weather.get("latitude")) if weather.get("latitude") is not None else None,
                longitude=float(weather.get("longitude")) if weather.get("longitude") is not None else None,
            )
            result["current_prediction"] = current
        except Exception as exc:
            result["current_prediction_error"] = str(exc)
    return result


@router.post("/ml/flood/predict", tags=["ml"])
def ml_flood_predict(body: FloodPredictRequest):
    return predict_flood(
        state=body.state,
        rainfall_mm=body.rainfall_24h_mm,
        month=body.month,
        duration_days=body.duration_days,
        affected_districts=body.affected_districts,
        affected_states=body.affected_states,
        main_cause=body.main_cause,
        latitude=body.latitude,
        longitude=body.longitude,
    )


@router.post("/ml/train/flood", tags=["ml"])
def ml_train_flood():
    from ml.flood_ml import train_flood_model
    return train_flood_model()


@router.get("/crisis", tags=["crisis"])
def list_crisis():
    return {"crises": [c.model_dump(mode="json") for c in crisis_service.list_crises()]}


@router.get("/crisis/state", tags=["crisis"])
def crisis_state(disaster_type: str | None = Query(default=None)):
    return crisis_service.get_crisis_state(disaster_type)


@router.get("/crisis/{disaster_type}", tags=["crisis"])
def crisis_by_type(disaster_type: str):
    parse_disaster(disaster_type)
    world = crisis_service.get_world(disaster_type)
    apply_risk_to_world(world)
    dash = crisis_tools.dashboard_summary(disaster_type)
    return {
        "crisis": world.crisis.model_dump(mode="json"),
        "metrics": dash["metrics"],
        "dashboard": dash,
        "incidents": [i.model_dump(mode="json") for i in world.incidents],
        "affected_zones": crisis_tools.affected_zones(disaster_type),
        "roads": [r.model_dump(mode="json") for r in world.roads.values()],
        "shelters": [s.model_dump(mode="json") for s in world.shelters.values()],
        "hospitals": [h.model_dump(mode="json") for h in world.hospitals.values()],
        "resources": [r.model_dump(mode="json") for r in world.resources.values()],
        "alerts": dash["current_alerts"],
        "evacuation": {
            "progress": dash["evacuation_progress"],
            "people_requiring_evacuation": dash["people_requiring_evacuation"],
            "people_evacuated_modeled": dash["people_evacuated_modeled"],
        },
        "simulation": crisis_tools.simulate_evacuation_tool(disaster_type),
    }


@router.get("/map/state", tags=["map"])
def map_state(disaster_type: str = Query(default="flood")):
    parse_disaster(disaster_type)
    return crisis_tools.map_state(disaster_type)


@router.get("/map/flood", tags=["map"], include_in_schema=True)
def map_flood():
    return crisis_tools.map_state("flood")


@router.get("/map/cyclone", tags=["map"], include_in_schema=True)
def map_cyclone():
    return crisis_tools.map_state("cyclone")


@router.get("/map/earthquake", tags=["map"], include_in_schema=True)
def map_earthquake():
    return crisis_tools.map_state("earthquake")


@router.get("/map/{disaster_type}", tags=["map"])
def map_by_type(disaster_type: str):
    parse_disaster(disaster_type)
    return crisis_tools.map_state(disaster_type)


@router.get("/incidents", tags=["entities"])
def incidents(disaster_type: str = Query(default="flood")):
    return {"incidents": [i.model_dump(mode="json") for i in crisis_service.get_incidents(disaster_type)]}


@router.get("/zones", tags=["entities"])
def zones(disaster_type: str = Query(default="flood")):
    world = crisis_service.get_world(disaster_type)
    apply_risk_to_world(world)
    return {"zones": [z.model_dump(mode="json") for z in world.zones.values()]}


@router.get("/roads", tags=["entities"])
def roads(disaster_type: str = Query(default="flood")):
    world = crisis_service.get_world(disaster_type)
    return {"roads": [r.model_dump(mode="json") for r in world.roads.values()]}


@router.get("/resources", tags=["entities"])
def resources(disaster_type: str = Query(default="flood")):
    world = crisis_service.get_world(disaster_type)
    return {"resources": [r.model_dump(mode="json") for r in world.resources.values()]}


@router.get("/hospitals", tags=["entities"])
def hospitals(disaster_type: str = Query(default="flood")):
    world = crisis_service.get_world(disaster_type)
    return {"hospitals": [h.model_dump(mode="json") for h in world.hospitals.values()]}


@router.get("/shelters", tags=["entities"])
def shelters(disaster_type: str = Query(default="flood")):
    world = crisis_service.get_world(disaster_type)
    return {"shelters": [s.model_dump(mode="json") for s in world.shelters.values()]}


@router.get("/risk", tags=["simulation"])
def risk(disaster_type: str = Query(default="flood"), zone_id: str | None = None):
    world = crisis_service.get_world(disaster_type)
    if zone_id:
        zone = crisis_service.get_zone_status(disaster_type, zone_id)
        return calculate_zone_risk(world, zone).model_dump(mode="json")
    return {"results": [r.model_dump(mode="json") for r in apply_risk_to_world(world)]}


@router.get("/routes", tags=["simulation"])
def routes(
    origin: str = Query(...),
    destination: str = Query(...),
    disaster_type: str = Query(default="flood"),
):
    world = crisis_service.get_world(disaster_type)
    results = find_safe_routes(world, origin, destination)
    return {"routes": [r.model_dump(mode="json") for r in results]}


@router.post("/routes", tags=["simulation"])
def routes_post(body: RouteQuery):
    world = crisis_service.get_world(body.disaster_type)
    results = find_safe_routes(world, body.origin, body.destination)
    return {"routes": [r.model_dump(mode="json") for r in results]}


@router.get("/geo/nearest-shelter", tags=["geo"])
def nearest_shelter(zone_id: str, disaster_type: str = "flood"):
    return crisis_tools.find_nearest_shelter(zone_id, disaster_type)


@router.get("/geo/nearest-hospital", tags=["geo"])
def nearest_hospital(zone_id: str, disaster_type: str = "flood"):
    return crisis_tools.find_nearest_hospital(zone_id, disaster_type)


@router.get("/geo/nearest-resource", tags=["geo"])
def nearest_resource(zone_id: str, disaster_type: str = "flood", resource_type: str | None = None):
    return crisis_tools.find_nearest_resource(zone_id, disaster_type, resource_type)


@router.post("/geo/within-radius", tags=["geo"])
def within_radius(body: GeoQuery):
    return {
        "resources": crisis_tools.resources_within_radius(
            body.latitude, body.longitude, body.radius_km, body.disaster_type
        ),
        "hospitals": crisis_tools.hospitals_within_radius(
            body.latitude, body.longitude, body.radius_km, body.disaster_type
        ),
        "shelters": crisis_tools.shelters_within_radius(
            body.latitude, body.longitude, body.radius_km, body.disaster_type
        ),
    }


@router.get("/alerts", tags=["alerts"])
def alerts(disaster_type: str = Query(default="flood")):
    world = crisis_service.get_world(disaster_type)
    return {"alerts": [a.model_dump(mode="json") for a in generate_alerts(world)]}


@router.get("/dashboard/summary", tags=["dashboard"])
def dashboard(disaster_type: str = Query(default="flood")):
    return crisis_tools.dashboard_summary(disaster_type)


@router.post("/events", tags=["events"])
def ingest_event(body: EventRequest):
    return crisis_service.apply_event(body.model_dump(exclude_none=True))


@router.post("/simulate", tags=["simulation"])
def simulate(body: SimulateRequest):
    parse_disaster(body.disaster_type)
    plan = body.plan or ResponsePlan(
        id=f"plan_{uuid4().hex[:8]}",
        title=body.title,
        objective=body.objective,
        actions=body.actions,
    )
    if not plan.id:
        plan.id = f"plan_{uuid4().hex[:8]}"
    world = crisis_service.get_world(body.disaster_type)
    outcome = simulate_plan(world, plan)
    return {
        "plan": outcome["plan"].model_dump(mode="json"),
        "simulation": outcome["simulation"].model_dump(mode="json"),
        "notes": outcome["notes"],
    }


@router.post("/simulate/compare", tags=["simulation"])
def simulate_compare(body: CompareRequest):
    parse_disaster(body.disaster_type)
    world = crisis_service.get_world(body.disaster_type)
    ranking = compare_plans(world, body.plans)
    return {"ranking": ranking}


@router.post("/ai/analyze", tags=["ai"])
def ai_analyze(body: AnalyzeRequest):
    analysis = analyst_agent.analyze(body.disaster_type, body.question)
    return analysis.model_dump(mode="json")


@router.post("/ai/plan", tags=["ai"])
def ai_plan(body: PlanRequest):
    return planner_agent.plan(body.disaster_type, body.objective)


@router.post("/ai/what-if", tags=["ai"])
def ai_what_if(body: WhatIfRequest):
    return what_if_agent.run(body.disaster_type, body.question).model_dump(mode="json")


@router.post("/ai/critique", tags=["ai"])
def ai_critique(body: SimulateRequest):
    plan = body.plan or ResponsePlan(title=body.title, objective=body.objective, actions=body.actions)
    return critic_agent.critique(body.disaster_type, plan).model_dump(mode="json")


@router.post("/scenarios", tags=["scenarios"])
def create_scenario(body: ScenarioCreateRequest):
    raw = scenario_service.run_scenario(body.disaster_type, body.description, body.changes)
    return {
        "scenario": raw["scenario"].model_dump(mode="json"),
        "affected_zones": raw["affected_zones"],
        "changed_risk": raw["changed_risk"],
        "changed_evacuation": raw["changed_evacuation"],
        "changed_resource_allocation": raw["changed_resource_allocation"],
        "changed_routes": raw["changed_routes"],
    }


@router.get("/scenarios", tags=["scenarios"])
def list_scenarios(disaster_type: str | None = None):
    return {"scenarios": [s.model_dump(mode="json") for s in scenario_service.list_scenarios(disaster_type)]}
