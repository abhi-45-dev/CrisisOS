from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
from shapely.geometry import LineString, Point, Polygon, shape

from app.geo import haversine_km
from app.providers.base import BaseProvider, ProviderStatus, SourceMetadata


class RoutingProfile(str, Enum):
    FASTEST = "fastest"
    BALANCED = "balanced"
    SAFEST = "safest"


class EmbeddedOsmRoutingProvider(BaseProvider):
    """In-process real OpenStreetMap road network routing engine for Tamil Nadu.

    Supports arbitrary WGS84 coordinates inside Tamil Nadu, snapping to the
    nearest road network junction and calculating FASTEST, BALANCED, and SAFEST
    paths with documented mathematical edge weighting.
    """

    def __init__(self, data_path: Path | None = None):
        self._data_path = data_path or (
            Path(__file__).resolve().parent.parent / "data" / "tamil_nadu_roads.json"
        )
        self._graph = nx.DiGraph()
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._load_network()

    def _load_network(self) -> None:
        if not self._data_path.exists():
            raise FileNotFoundError(f"Road network file missing: {self._data_path}")

        with self._data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self._nodes = data.get("nodes", {})
        self._edges = data.get("edges", [])

        # Build bidirectional NetworkX graph
        for node_id, node in self._nodes.items():
            self._graph.add_node(
                node_id,
                latitude=node["latitude"],
                longitude=node["longitude"],
                name=node["name"],
                district=node.get("district", ""),
            )

        for edge in self._edges:
            u, v = edge["u"], edge["v"]
            dist_km = float(edge["distance_km"])
            speed_kmh = float(edge["speed_kmh"])
            travel_time_min = (dist_km / max(speed_kmh, 10.0)) * 60.0

            # Forward edge
            self._graph.add_edge(
                u,
                v,
                id=edge["id"],
                name=edge["name"],
                road_class=edge.get("road_class", "primary"),
                distance_km=dist_km,
                speed_kmh=speed_kmh,
                travel_time_min=travel_time_min,
            )
            # Reverse edge (two-way roads)
            self._graph.add_edge(
                v,
                u,
                id=f"{edge['id']}_rev",
                name=edge["name"],
                road_class=edge.get("road_class", "primary"),
                distance_km=dist_km,
                speed_kmh=speed_kmh,
                travel_time_min=travel_time_min,
            )

    @property
    def name(self) -> str:
        return "OpenStreetMap / Geofabrik"

    @property
    def dataset_name(self) -> str:
        return "Tamil Nadu Highway & Arterial Road Graph"

    def get_metadata(self) -> SourceMetadata:
        return SourceMetadata(
            provider=self.name,
            dataset=self.dataset_name,
            source_url="https://download.geofabrik.de/asia/india/southern-zone.html",
            source_id="OSM-TN-HIGHWAYS-2026",
            retrieved_at=datetime(2026, 1, 20, 0, 0, tzinfo=timezone.utc),
            license="ODbL 1.0",
            status=ProviderStatus.READY,
            notes=(
                f"Active in-process road graph with {self._graph.number_of_nodes()} junctions "
                f"and {self._graph.number_of_edges()} road links across Tamil Nadu."
            ),
        )

    def snap_to_nearest_node(self, lat: float, lon: float) -> tuple[str, float]:
        """Find the nearest road network node to a given coordinate."""
        nearest_node = None
        min_dist = float("inf")
        for node_id, node in self._nodes.items():
            dist = haversine_km(lat, lon, node["latitude"], node["longitude"])
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
        return nearest_node, min_dist

    def _calculate_edge_hazard(
        self,
        u: str,
        v: str,
        hazard_polygons: list[dict[str, Any]] | None,
    ) -> float:
        """Compute the maximum flood hazard probability intersecting an edge."""
        if not hazard_polygons:
            return 0.0

        u_node = self._nodes[u]
        v_node = self._nodes[v]
        edge_geom = LineString([(u_node["longitude"], u_node["latitude"]), (v_node["longitude"], v_node["latitude"])])

        max_hazard = 0.0
        for hp in hazard_polygons:
            poly_geom = shape(hp["geometry"])
            if edge_geom.intersects(poly_geom):
                max_hazard = max(max_hazard, float(hp.get("flood_probability", 0.0)))

        return max_hazard

    def calculate_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        profile: RoutingProfile = RoutingProfile.FASTEST,
        hazard_polygons: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Calculate route over real OSM network between arbitrary coordinates."""
        start_node, start_snap_km = self.snap_to_nearest_node(origin_lat, origin_lon)
        end_node, end_snap_km = self.snap_to_nearest_node(dest_lat, dest_lon)

        if start_node == end_node:
            direct_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
            return {
                "feasible": True,
                "profile": profile.value,
                "total_distance_km": round(direct_km, 2),
                "estimated_duration_min": round((direct_km / 30.0) * 60, 1),
                "path_nodes": [start_node],
                "road_names": ["Direct Local Road Access"],
                "coordinates": [
                    [round(origin_lon, 5), round(origin_lat, 5)],
                    [round(dest_lon, 5), round(dest_lat, 5)],
                ],
                "warnings": [],
            }

        # Build edge weight function based on documented routing policy
        def edge_weight(u, v, d):
            base_time = d["travel_time_min"]
            hazard = self._calculate_edge_hazard(u, v, hazard_polygons)

            if profile == RoutingProfile.FASTEST:
                return base_time
            elif profile == RoutingProfile.BALANCED:
                # Moderate detour penalty
                return base_time * (1.0 + 3.5 * hazard)
            elif profile == RoutingProfile.SAFEST:
                # Strong exponential penalty for high flood hazard
                if hazard >= 0.80:
                    return base_time * 50.0  # Strongly avoid critical hazard
                return base_time * (1.0 + 15.0 * (hazard ** 2))
            return base_time

        try:
            path = nx.dijkstra_path(self._graph, start_node, end_node, weight=edge_weight)
        except nx.NetworkXNoPath:
            return {
                "feasible": False,
                "profile": profile.value,
                "total_distance_km": 0.0,
                "estimated_duration_min": 0.0,
                "path_nodes": [],
                "road_names": [],
                "coordinates": [],
                "warnings": ["NO ROUTE CONFIDENTLY RECOMMENDED: No reachable path on network."],
            }

        # Build full geometry and metrics
        coords = [[round(origin_lon, 5), round(origin_lat, 5)]]
        road_names = []
        total_dist = start_snap_km
        total_time = (start_snap_km / 35.0) * 60.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self._graph[u][v]
            total_dist += edge_data["distance_km"]
            total_time += edge_data["travel_time_min"]
            road_names.append(edge_data["name"])

            u_node = self._nodes[u]
            v_node = self._nodes[v]
            coords.append([round(u_node["longitude"], 5), round(u_node["latitude"], 5)])
            coords.append([round(v_node["longitude"], 5), round(v_node["latitude"], 5)])

        coords.append([round(dest_lon, 5), round(dest_lat, 5)])
        total_dist += end_snap_km
        total_time += (end_snap_km / 35.0) * 60.0

        # Remove consecutive duplicate points
        cleaned_coords = []
        for pt in coords:
            if not cleaned_coords or cleaned_coords[-1] != pt:
                cleaned_coords.append(pt)

        # Deduplicate sequential road names
        dedup_roads = []
        for r in road_names:
            if not dedup_roads or dedup_roads[-1] != r:
                dedup_roads.append(r)

        return {
            "feasible": True,
            "profile": profile.value,
            "total_distance_km": round(total_dist, 2),
            "estimated_duration_min": round(total_time, 1),
            "path_nodes": path,
            "road_names": dedup_roads,
            "coordinates": cleaned_coords,
            "geometry": {
                "type": "LineString",
                "coordinates": cleaned_coords,
            },
            "warnings": [],
        }


osm_routing_provider = EmbeddedOsmRoutingProvider()
