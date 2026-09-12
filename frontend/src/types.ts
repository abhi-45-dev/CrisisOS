export interface SourceMetadata {
  provider: string;
  dataset: string;
  source_url: string;
  source_id: string;
  retrieved_at: string;
  license: string;
  status: "READY" | "DEGRADED" | "DOWN" | "PENDING";
  item_count?: number;
  notes?: string;
}

export interface SystemEvidence {
  operational_handcrafted_data_count: number;
  product_name: string;
  geographic_scope: string;
  timestamp: string;
  sources: Record<string, SourceMetadata>;
  model: {
    name: string;
    algorithm: string;
    metrics: {
      pr_auc: number;
      roc_auc: number;
      brier_score: number;
      precision: number;
      recall: number;
      f1: number;
      accuracy: number;
      test_samples: number;
    } | null;
    training_manifest: {
      dataset: string;
      dataset_doi: string;
      total_rows: number;
      positives: number;
      weak_negatives: number;
      features: string[];
      target: string;
    } | null;
    calibrated: boolean;
  };
}

export interface Hospital {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  district?: string;
  pincode?: string;
  emergency_services?: boolean;
  available_beds: null | number;
  source: string;
  facility_type: "hospital";
  distance_km?: number;
}

export interface EvacuationFacility {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  district?: string;
  facility_category: "official_cyclone_shelter" | "candidate_evacuation_facility";
  is_official: boolean;
  capacity?: number;
  source: string;
  facility_type: "shelter";
  distance_km?: number;
}

export interface HazardProperties {
  cell_id: string;
  flood_probability: number;
  risk_band: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  observed_rain_24h_mm: number;
  soil_moisture: number;
  model_version: string;
  inference_timestamp: string;
  local_explanations: Array<{
    feature: string;
    importance_weight: number;
    value: number;
    direction: "increases_risk" | "decreases_risk" | "neutral";
  }>;
  baseline_flood_probability?: number;
  scenario_flood_probability?: number;
  delta_flood_probability?: number;
  risk_band_escalated?: boolean;
}

export interface HazardFeature {
  type: "Feature";
  id: string;
  properties: HazardProperties;
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
}

export interface HazardFeatureCollection {
  type: "FeatureCollection";
  metadata: {
    generated_at: string;
    cell_count: number;
    model_version: string;
    scenario_override_mm?: number | null;
  };
  features: HazardFeature[];
}

export interface CoordinatePrediction {
  flood_probability: number;
  risk_band: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  model_version: string;
  inference_timestamp: string;
  local_explanations: Array<{
    feature: string;
    importance_weight: number;
    value: number;
    direction: "increases_risk" | "decreases_risk" | "neutral";
  }>;
  coordinates: [number, number];
  weather_inputs: {
    rain_24h_mm: number;
    rain_72h_mm: number;
    soil_moisture: number;
    weather_source: string;
    observed_at?: string | null;
  };
}

export interface HazardSegment {
  segment_index: number;
  coordinates: [number, number][];
  distance_km: number;
  flood_probability: number;
  risk_band: "low" | "moderate" | "high" | "critical";
}

export interface RouteExposure {
  profile: "fastest" | "balanced" | "safest";
  feasible: boolean;
  total_distance_km: number;
  baseline_network_duration_min: number;
  max_flood_probability: number;
  mean_flood_probability: number;
  high_risk_distance_km: number;
  critical_risk_distance_km: number;
  risk_verdict: "SAFE" | "MODERATE_EXPOSURE" | "ELEVATED_EXPOSURE" | "CRITICAL_HAZARD" | "NO_ROUTE";
  road_names: string[];
  hazard_segments: HazardSegment[];
  geometry: {
    type: "LineString";
    coordinates: [number, number][];
  };
  warnings: string[];
}

export interface RouteCompareResponse {
  origin: [number, number];
  destination: [number, number];
  snapped_origin: [number, number];
  snapped_destination: [number, number];
  fastest: RouteExposure;
  balanced: RouteExposure;
  safest: RouteExposure;
  comparison_summary: string;
  recommended_profile: "fastest" | "balanced" | "safest";
  evaluation_notes: string[];
}

export interface CandidateOption {
  facility: Hospital | EvacuationFacility;
  destination_hazard: CoordinatePrediction;
  fastest_route: RouteExposure;
  safest_route: RouteExposure;
  penalty_score: number;
}

export interface SurvivorRecommendation {
  survivor_location: [number, number];
  survivor_hazard: CoordinatePrediction;
  recommended_destination: (Hospital | EvacuationFacility) | null;
  recommended_route?: RouteExposure;
  fastest_alternative_route?: RouteExposure;
  candidate_options: CandidateOption[];
  verdict: "RECOMMENDED" | "NO ROUTE CONFIDENTLY RECOMMENDED" | "NO_FACILITY_WITHIN_RANGE";
  verdict_ta?: string | null;
  rationale: string;
  rationale_ta?: string | null;
  disclaimer: string;
}

export interface ScenarioSimulationResponse {
  scenario: {
    type: string;
    rainfall_delta_mm: number;
    model_version: string;
    summary: string;
    metrics: {
      total_cells_evaluated: number;
      escalated_cells_count: number;
      critically_exposed_cells: number;
      average_probability_delta: number;
      maximum_probability_delta: number;
    };
  };
  route_impact: Record<string, unknown> | null;
  geojson: HazardFeatureCollection;
}

export interface CitizenIncident {
  id: string;
  latitude: number;
  longitude: number;
  incident_type: string;
  description: string;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  status: "UNVERIFIED" | "VERIFIED_BY_OFFICIALS" | "RESOLVED";
  verification_source?: string | null;
  reported_by: string;
  created_at: string;
  distance_km?: number;
}
