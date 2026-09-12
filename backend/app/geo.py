from math import atan2, cos, radians, sin, sqrt

from app.models.location import Hospital, Road, Shelter, Zone
from app.models.resource import Resource


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    return ((lat1 + lat2) / 2, (lon1 + lon2) / 2)


def nearest(
    lat: float,
    lon: float,
    items: list[Zone] | list[Hospital] | list[Shelter] | list[Resource],
    *,
    limit: int = 1,
    radius_km: float | None = None,
) -> list[tuple[object, float]]:
    scored: list[tuple[object, float]] = []
    for item in items:
        dist = haversine_km(lat, lon, item.latitude, item.longitude)
        if radius_km is not None and dist > radius_km:
            continue
        scored.append((item, dist))
    scored.sort(key=lambda pair: pair[1])
    return scored[:limit]


def items_within_radius(
    lat: float,
    lon: float,
    items: list,
    radius_km: float,
) -> list[tuple[object, float]]:
    return nearest(lat, lon, items, limit=len(items), radius_km=radius_km)
