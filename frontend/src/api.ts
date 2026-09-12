import type {
  BackendRouteCompareResponse,
  CitizenIncident,
  CoordinatePrediction,
  EvacuationFacility,
  HazardFeatureCollection,
  Hospital,
  HospitalListResponse,
  LocalExplanation,
  NearbyInfrastructureResponse,
  NormalizedNearbyInfrastructure,
  RouteCompareResponse,
  ScenarioSimulationResponse,
  ShelterListResponse,
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

function normalizeExplanation(e: any): LocalExplanation {
  const isRisk =
    e.impact === "increases_hazard" ||
    e.impact === "increases_hazard_lowland" ||
    e.impact === "near_river_channel" ||
    e.direction === "increases_risk";

  return {
    feature: e.feature || "unknown",
    value: e.observed_value ?? e.value ?? 0,
    observed_value: e.observed_value ?? e.value ?? 0,
    baseline_mean: e.baseline_mean ?? 0,
    attribution_weight: e.attribution_weight ?? e.importance_weight ?? 0,
    importance_weight: e.attribution_weight ?? e.importance_weight ?? 0,
    impact: e.impact,
    direction: isRisk ? "increases_risk" : "decreases_risk",
  };
}

function normalizePrediction(p: any): CoordinatePrediction {
  return {
    ...p,
    risk_band: (p.risk_band || "low").toUpperCase() as any,
    local_explanations: Array.isArray(p.local_explanations)
      ? p.local_explanations.map(normalizeExplanation)
      : [],
    coordinates: p.coordinates || [0, 0],
    weather_inputs: p.weather_inputs || {
      rain_24h_mm: 0,
      rain_72h_mm: 0,
      soil_moisture: 0,
      weather_source: "Open-Meteo",
    },
  };
}

export const api = {
  // System Health & Provenance Evidence
  health: async () =>
    request<{ status: string; system: string; providers: Record<string, unknown> }>("/api/v1/health"),

  evidence: async (): Promise<SystemEvidence> => request<SystemEvidence>("/api/v1/evidence"),

  dataSources: async () =>
    request<{ timestamp: string; sources: Record<string, SourceMetadata> }>("/api/v1/data-sources"),

  // Hazard Surface & ML Inference
  hazardGrid: async (rainfallOverrideMm?: number, forceRefresh?: boolean): Promise<HazardFeatureCollection> => {
    const raw = await request<HazardFeatureCollection>(
      `/api/v1/hazard/grid${query({ rainfall_override_mm: rainfallOverrideMm, force_refresh: forceRefresh })}`
    );
    if (!raw || !Array.isArray(raw.features)) {
      return { type: "FeatureCollection", metadata: { generated_at: "", cell_count: 0, model_version: "" }, features: [] };
    }
    // Normalize risk_band casing and explanations on each feature
    const normalizedFeatures = raw.features.map((f) => ({
      ...f,
      properties: {
        ...f.properties,
        risk_band: (f.properties.risk_band || "low").toUpperCase() as any,
        local_explanations: Array.isArray(f.properties.local_explanations)
          ? f.properties.local_explanations.map(normalizeExplanation)
          : [],
      },
    }));
    return {
      ...raw,
      features: normalizedFeatures,
    };
  },

  predictCoordinate: async (
    latitude: number,
    longitude: number,
    rainfallOverrideMm?: number
  ): Promise<CoordinatePrediction> => {
    const raw = await request<any>("/api/v1/hazard/predict", {
      method: "POST",
      body: JSON.stringify({
        latitude,
        longitude,
        rainfall_override_mm: rainfallOverrideMm,
      }),
    });
    return normalizePrediction(raw);
  },

  // Infrastructure: Hospitals & Evacuation Facilities (Normalized from wrapper objects)
  nearbyInfrastructure: async (
    lat: number,
    lon: number,
    radiusKm = 25.0,
    facilityType: "all" | "hospital" | "shelter" = "all",
    limit = 10
  ): Promise<NormalizedNearbyInfrastructure> => {
    try {
      const resp = await request<NearbyInfrastructureResponse>(
        `/api/v1/infrastructure/nearby${query({ lat, lon, radius_km: radiusKm, facility_type: facilityType, limit })}`
      );
      return {
        latitude: resp?.query?.latitude ?? lat,
        longitude: resp?.query?.longitude ?? lon,
        radius_km: resp?.query?.radius_km ?? radiusKm,
        hospitals: Array.isArray(resp?.hospitals) ? resp.hospitals : [],
        evacuation_facilities: Array.isArray(resp?.shelters) ? resp.shelters : [],
      };
    } catch (err) {
      console.warn("nearbyInfrastructure request failed:", err);
      return {
        latitude: lat,
        longitude: lon,
        radius_km: radiusKm,
        hospitals: [],
        evacuation_facilities: [],
      };
    }
  },

  hospitals: async (district?: string): Promise<Hospital[]> => {
    try {
      const resp = await request<HospitalListResponse | Hospital[]>(
        `/api/v1/infrastructure/hospitals${query({ district })}`
      );
      if (Array.isArray(resp)) return resp;
      return Array.isArray(resp?.hospitals) ? resp.hospitals : [];
    } catch (err) {
      console.warn("Failed to load hospitals list:", err);
      return [];
    }
  },

  shelters: async (officialOnly?: boolean): Promise<EvacuationFacility[]> => {
    try {
      const resp = await request<ShelterListResponse | EvacuationFacility[]>(
        `/api/v1/infrastructure/shelters${query({ official_only: officialOnly })}`
      );
      if (Array.isArray(resp)) return resp;
      return Array.isArray(resp?.shelters) ? resp.shelters : [];
    } catch (err) {
      console.warn("Failed to load shelters list:", err);
      return [];
    }
  },

  // Road Routing & Exposure Comparison (Aligned schema & normalized response)
  compareRoutes: async (
    originLat: number,
    originLon: number,
    destLat: number,
    destLon: number,
    hazardPolygons: any[] = []
  ): Promise<RouteCompareResponse> => {
    const resp = await request<BackendRouteCompareResponse>("/api/v1/routes/compare", {
      method: "POST",
      body: JSON.stringify({
        origin: { latitude: originLat, longitude: originLon },
        destination: { latitude: destLat, longitude: destLon },
        hazard_polygons: hazardPolygons,
      }),
    });
    return {
      origin: resp.origin,
      destination: resp.destination,
      fastest: resp.routes?.fastest,
      balanced: resp.routes?.balanced,
      safest: resp.routes?.safest,
      comparison_summary: resp.recommendation?.rationale || "",
      verdict: resp.recommendation?.verdict || "",
      recommended_profile: resp.recommendation?.selected_profile || "safest",
    };
  },

  // Survivor Decision Support
  recommendEvacuation: async (body: {
    latitude: number;
    longitude: number;
    preferred_type?: "all" | "hospital" | "shelter";
    language?: "en" | "ta";
    radius_km?: number;
  }): Promise<SurvivorRecommendation> => {
    const resp = await request<SurvivorRecommendation>("/api/v1/survivor/recommend", {
      method: "POST",
      body: JSON.stringify(body),
    });
    if (resp?.survivor_hazard) {
      resp.survivor_hazard = normalizePrediction(resp.survivor_hazard);
    }
    if (Array.isArray(resp?.candidate_options)) {
      resp.candidate_options = resp.candidate_options.map((cand) => ({
        ...cand,
        destination_hazard: normalizePrediction(cand.destination_hazard),
      }));
    }
    return resp;
  },

  // Scenario Simulation (What-If rainfall)
  simulateScenario: async (rainfallDeltaMm: number): Promise<ScenarioSimulationResponse> => {
    const raw = await request<ScenarioSimulationResponse>("/api/v1/scenarios/rainfall", {
      method: "POST",
      body: JSON.stringify({ rainfall_delta_mm: rainfallDeltaMm }),
    });
    if (raw?.geojson && Array.isArray(raw.geojson.features)) {
      raw.geojson.features = raw.geojson.features.map((f) => ({
        ...f,
        properties: {
          ...f.properties,
          risk_band: (f.properties.risk_band || "low").toUpperCase() as any,
          local_explanations: Array.isArray(f.properties.local_explanations)
            ? f.properties.local_explanations.map(normalizeExplanation)
            : [],
        },
      }));
    }
    return raw;
  },

  // Citizen Incident Reports
  reportIncident: async (body: {
    latitude: number;
    longitude: number;
    incident_type: string;
    description: string;
    severity?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
    reporter_name?: string;
  }): Promise<CitizenIncident> => {
    return request<CitizenIncident>("/api/v1/incidents", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  listIncidents: async (status?: string, severity?: string): Promise<CitizenIncident[]> => {
    try {
      const resp = await request<CitizenIncident[]>(`/api/v1/incidents${query({ status, severity })}`);
      return Array.isArray(resp) ? resp : [];
    } catch (err) {
      console.warn("Failed to load incidents list:", err);
      return [];
    }
  },

  nearbyIncidents: async (lat: number, lon: number, radiusKm = 25.0): Promise<CitizenIncident[]> => {
    try {
      const resp = await request<CitizenIncident[]>(`/api/v1/incidents/nearby${query({ lat, lon, radius_km: radiusKm })}`);
      return Array.isArray(resp) ? resp : [];
    } catch (err) {
      console.warn("Failed to load nearby incidents:", err);
      return [];
    }
  },
};

export { API_BASE };
