import type {
  CitizenIncident,
  CoordinatePrediction,
  EvacuationFacility,
  HazardFeatureCollection,
  Hospital,
  RouteCompareResponse,
  ScenarioSimulationResponse,
  SourceMetadata,
  SurvivorRecommendation,
  SystemEvidence,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}${body ? ` — ${body}` : ""}`);
  }

  return response.json() as Promise<T>;
}

const query = (params: Record<string, string | number | boolean | undefined>) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) search.set(key, String(value));
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
};

export const api = {
  // System Health & Provenance Evidence
  health: () => request<{ status: string; system: string; providers: Record<string, unknown> }>("/api/v1/health"),
  evidence: () => request<SystemEvidence>("/api/v1/evidence"),
  dataSources: () => request<{ timestamp: string; sources: Record<string, SourceMetadata> }>("/api/v1/data-sources"),

  // Hazard Surface & ML Inference
  hazardGrid: (rainfallOverrideMm?: number, forceRefresh?: boolean) =>
    request<HazardFeatureCollection>(
      `/api/v1/hazard/grid${query({ rainfall_override_mm: rainfallOverrideMm, force_refresh: forceRefresh })}`
    ),

  predictCoordinate: (latitude: number, longitude: number, rainfallOverrideMm?: number) =>
    request<CoordinatePrediction>("/api/v1/hazard/predict", {
      method: "POST",
      body: JSON.stringify({
        latitude,
        longitude,
        rainfall_override_mm: rainfallOverrideMm,
      }),
    }),

  // Infrastructure: Hospitals & Evacuation Facilities
  nearbyInfrastructure: (lat: number, lon: number, radiusKm = 25.0) =>
    request<{
      center: [number, number];
      radius_km: number;
      hospitals: Hospital[];
      evacuation_facilities: EvacuationFacility[];
    }>(`/api/v1/infrastructure/nearby${query({ lat, lon, radius_km: radiusKm })}`),

  hospitals: (lat?: number, lon?: number, radiusKm?: number) =>
    request<Hospital[]>(`/api/v1/infrastructure/hospitals${query({ lat, lon, radius_km: radiusKm })}`),

  shelters: (lat?: number, lon?: number, radiusKm?: number) =>
    request<EvacuationFacility[]>(`/api/v1/infrastructure/shelters${query({ lat, lon, radius_km: radiusKm })}`),

  // Road Routing & Exposure Comparison
  compareRoutes: (originLat: number, originLon: number, destLat: number, destLon: number) =>
    request<RouteCompareResponse>("/api/v1/routes/compare", {
      method: "POST",
      body: JSON.stringify({
        origin_lat: originLat,
        origin_lon: originLon,
        dest_lat: destLat,
        dest_lon: destLon,
      }),
    }),

  // Survivor Decision Support
  recommendEvacuation: (body: {
    latitude: number;
    longitude: number;
    preferred_type?: "all" | "hospital" | "shelter";
    language?: "en" | "ta";
    radius_km?: number;
  }) =>
    request<SurvivorRecommendation>("/api/v1/survivor/recommend", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Scenario Simulation (What-If rainfall)
  simulateScenario: (rainfallDeltaMm: number) =>
    request<ScenarioSimulationResponse>("/api/v1/scenarios/rainfall", {
      method: "POST",
      body: JSON.stringify({ rainfall_delta_mm: rainfallDeltaMm }),
    }),

  // Citizen Incident Reports
  reportIncident: (body: {
    latitude: number;
    longitude: number;
    incident_type: string;
    description: string;
    severity?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
    reporter_name?: string;
  }) =>
    request<CitizenIncident>("/api/v1/incidents", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listIncidents: (status?: string, severity?: string) =>
    request<CitizenIncident[]>(`/api/v1/incidents${query({ status, severity })}`),

  nearbyIncidents: (lat: number, lon: number, radiusKm = 25.0) =>
    request<CitizenIncident[]>(`/api/v1/incidents/nearby${query({ lat, lon, radius_km: radiusKm })}`),
};

export { API_BASE };
