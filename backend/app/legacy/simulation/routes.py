from heapq import heappop, heappush

from app.exceptions import NotFoundError
from app.models.location import RoadStatus
from app.models.response import RouteLeg, RouteResult
from app.world import CrisisWorld

BLOCKED = RoadStatus.BLOCKED
DANGEROUS_PENALTY = 45.0
CONGESTED_MULTIPLIER = 2.2
DANGEROUS_MULTIPLIER = 3.5


def road_cost(road) -> float | None:
    if road.status == BLOCKED:
        return None
    cost = float(road.travel_time_min)
    if road.status == RoadStatus.CONGESTED:
        cost *= CONGESTED_MULTIPLIER
    if road.status == RoadStatus.DANGEROUS:
        cost *= DANGEROUS_MULTIPLIER
        cost += DANGEROUS_PENALTY
    cost += float(road.risk_score) * 0.15
    return cost


def build_adjacency(world: CrisisWorld) -> dict[str, list[tuple[str, object, float]]]:
    graph: dict[str, list[tuple[str, object, float]]] = {zone_id: [] for zone_id in world.zones}
    for road in world.roads.values():
        cost = road_cost(road)
        if cost is None:
            continue
        if road.from_zone in graph and road.to_zone in graph:
            graph[road.from_zone].append((road.to_zone, road, cost))
            graph[road.to_zone].append((road.from_zone, road, cost))
    return graph


def find_safe_route(world: CrisisWorld, origin: str, destination: str) -> RouteResult:
    if origin not in world.zones:
        raise NotFoundError(f"Unknown origin zone '{origin}'", {"id": origin})
    if destination not in world.zones:
        raise NotFoundError(f"Unknown destination zone '{destination}'", {"id": destination})
    if origin == destination:
        return RouteResult(
            origin=origin,
            destination=destination,
            path_zones=[origin],
            total_distance_km=0,
            estimated_travel_time_min=0,
            route_risk=0,
            feasible=True,
        )

    graph = build_adjacency(world)
    dist: dict[str, float] = {origin: 0.0}
    prev: dict[str, tuple[str, object]] = {}
    heap: list[tuple[float, str]] = [(0.0, origin)]
    seen: set[str] = set()

    while heap:
        cost, node = heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        if node == destination:
            break
        for neighbor, road, edge_cost in graph.get(node, []):
            candidate = cost + edge_cost
            if candidate < dist.get(neighbor, float("inf")):
                dist[neighbor] = candidate
                prev[neighbor] = (node, road)
                heappush(heap, (candidate, neighbor))

    if destination not in prev:
        return RouteResult(
            origin=origin,
            destination=destination,
            feasible=False,
            warnings=[
                f"No safe route from {origin} to {destination}. Blocked or disconnected roads prevent travel."
            ],
        )

    zones = [destination]
    roads_used = []
    legs: list[RouteLeg] = []
    node = destination
    while node != origin:
        parent, road = prev[node]
        roads_used.append(road.id)
        legs.append(
            RouteLeg(
                road_id=road.id,
                road_name=road.name,
                from_zone=parent,
                to_zone=node,
                distance_km=road.distance_km,
                travel_time_min=road.travel_time_min,
                status=road.status.value,
            )
        )
        zones.append(parent)
        node = parent
    zones.reverse()
    legs.reverse()
    roads_used.reverse()

    warnings = []
    risk_sum = 0.0
    travel = 0.0
    distance = 0.0
    for road_id in roads_used:
        road = world.roads[road_id]
        risk_sum += road.risk_score
        travel += road.travel_time_min
        distance += road.distance_km
        if road.status == RoadStatus.DANGEROUS:
            warnings.append(f"{road.name} ({road.id}) is dangerous and was penalized but remains passable.")
        if road.status == RoadStatus.CONGESTED:
            warnings.append(f"{road.name} ({road.id}) is congested.")

    avg_risk = risk_sum / max(len(roads_used), 1)
    return RouteResult(
        origin=origin,
        destination=destination,
        path_zones=zones,
        roads_used=roads_used,
        legs=legs,
        total_distance_km=round(distance, 2),
        estimated_travel_time_min=round(travel, 2),
        route_risk=round(avg_risk, 2),
        warnings=warnings,
        feasible=True,
    )


def find_safe_routes(world: CrisisWorld, origin: str, destination: str) -> list[RouteResult]:
    primary = find_safe_route(world, origin, destination)
    alternatives: list[RouteResult] = [primary]
    if not primary.feasible or not primary.roads_used:
        return alternatives

    clone = world.clone()
    worst = max(primary.roads_used, key=lambda rid: clone.roads[rid].risk_score)
    clone.roads[worst].status = RoadStatus.BLOCKED
    alt = find_safe_route(clone, origin, destination)
    if alt.feasible and alt.roads_used != primary.roads_used:
        alt.warnings.append(f"Alternative avoids {worst}.")
        alternatives.append(alt)
    return alternatives
