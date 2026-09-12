import { useEffect, useRef, useState } from "react";
import type {
  CitizenIncident,
  EvacuationFacility,
  HazardFeature,
  HazardFeatureCollection,
  Hospital,
  RouteExposure,
} from "../types";

// Standard Leaflet typings for window.L
interface LeafletMapInstance {
  setView: (center: [number, number], zoom: number, options?: Record<string, unknown>) => LeafletMapInstance;
  fitBounds: (bounds: Array<[number, number]>, options?: Record<string, unknown>) => LeafletMapInstance;
  invalidateSize: (options?: Record<string, unknown>) => LeafletMapInstance;
  remove: () => void;
  on: (event: string, fn: (e: any) => void) => LeafletMapInstance;
}

interface LeafletLayer {
  addTo: (map: LeafletMapInstance) => LeafletLayer;
  bindPopup: (content: string, options?: Record<string, unknown>) => LeafletLayer;
  bindTooltip: (content: string, options?: Record<string, unknown>) => LeafletLayer;
}

interface LeafletLib {
  map: (element: HTMLElement, options?: Record<string, unknown>) => LeafletMapInstance;
  tileLayer: (url: string, options?: Record<string, unknown>) => LeafletLayer;
  geoJSON: (geojson: any, options?: Record<string, unknown>) => LeafletLayer;
  polyline: (latlngs: number[][], options?: Record<string, unknown>) => LeafletLayer;
  marker: (latlng: [number, number], options?: Record<string, unknown>) => LeafletLayer;
  circleMarker: (latlng: [number, number], options?: Record<string, unknown>) => LeafletLayer;
  layerGroup: (layers?: LeafletLayer[]) => any;
  divIcon: (options?: Record<string, unknown>) => unknown;
}

type LeafletWindow = Window & typeof globalThis & { L?: LeafletLib };

export interface LeafletCrisisMapProps {
  hazardGrid?: HazardFeatureCollection | null;
  hospitals?: Hospital[];
  shelters?: EvacuationFacility[];
  incidents?: CitizenIncident[];
  routes?: {
    fastest?: RouteExposure | null;
    safest?: RouteExposure | null;
  };
  activePin?: [number, number] | null;
  onMapClick?: (lat: number, lon: number) => void;
  onSelectCell?: (feature: HazardFeature) => void;
  focusDistrictCoords?: [number, number] | null;
}

function escapeHtml(str: string): string {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const RISK_COLORS: Record<string, string> = {
  LOW: "#10b981",
  MODERATE: "#f59e0b",
  HIGH: "#f97316",
  CRITICAL: "#ef4444",
};

export function LeafletCrisisMap({
  hazardGrid,
  hospitals = [],
  shelters = [],
  incidents = [],
  routes,
  activePin,
  onMapClick,
  onSelectCell,
  focusDistrictCoords,
}: LeafletCrisisMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<LeafletMapInstance | null>(null);

  // Layer groups
  const hazardLayerRef = useRef<any>(null);
  const hospitalLayerRef = useRef<any>(null);
  const shelterLayerRef = useRef<any>(null);
  const incidentLayerRef = useRef<any>(null);
  const routeLayerRef = useRef<any>(null);
  const pinLayerRef = useRef<any>(null);

  // Layer visibility state
  const [showHazard, setShowHazard] = useState(true);
  const [showHospitals, setShowHospitals] = useState(true);
  const [showShelters, setShowShelters] = useState(true);
  const [showIncidents, setShowIncidents] = useState(true);
  const [showRoutes, setShowRoutes] = useState(true);

  // Initialize Map
  useEffect(() => {
    const container = containerRef.current;
    const L = (window as LeafletWindow).L;
    if (!container || !L || mapInstanceRef.current) return;

    // Centered on Tamil Nadu (approx lat 11.1271, lon 78.6569)
    const map = L.map(container, {
      zoomControl: true,
      preferCanvas: true,
      attributionControl: true,
      minZoom: 6,
      maxZoom: 18,
    }).setView([11.1271, 78.6569], 7);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> | DisasterPulse TN',
    }).addTo(map);

    // Click handler on map
    map.on("click", (e: any) => {
      if (onMapClick && e.latlng) {
        onMapClick(Number(e.latlng.lat.toFixed(5)), Number(e.latlng.lng.toFixed(5)));
      }
    });

    hazardLayerRef.current = L.layerGroup().addTo(map);
    hospitalLayerRef.current = L.layerGroup().addTo(map);
    shelterLayerRef.current = L.layerGroup().addTo(map);
    incidentLayerRef.current = L.layerGroup().addTo(map);
    routeLayerRef.current = L.layerGroup().addTo(map);
    pinLayerRef.current = L.layerGroup().addTo(map);

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update Hazard Grid
  useEffect(() => {
    const L = (window as LeafletWindow).L;
    const group = hazardLayerRef.current;
    if (!L || !group) return;

    group.clearLayers();
    if (!showHazard || !hazardGrid || !hazardGrid.features) return;

    hazardGrid.features.forEach((feature) => {
      const p = feature.properties;
      const color = RISK_COLORS[p.risk_band] || "#10b981";
      const probPct = (p.flood_probability * 100).toFixed(1);

      // GeoJSON polygon coordinates in GeoJSON are [lon, lat], Leaflet polygon takes [lat, lon]
      const rings = feature.geometry.coordinates.map((ring) =>
        ring.map(([lon, lat]) => [lat, lon] as [number, number])
      );

      const poly = (L as any).polygon(rings, {
        color: color,
        weight: 1,
        opacity: 0.8,
        fillColor: color,
        fillOpacity: p.risk_band === "CRITICAL" ? 0.45 : p.risk_band === "HIGH" ? 0.35 : 0.2,
      });

      const explanations = (p.local_explanations || [])
        .slice(0, 3)
        .map((e) => `<div>• ${escapeHtml(e.feature)}: ${e.direction === "increases_risk" ? "▲" : "▼"}</div>`)
        .join("");

      const popupHtml = `
        <div style="font-family:Inter,sans-serif;font-size:12px;color:#111817;min-width:180px;">
          <div style="font-weight:700;font-size:13px;color:#0d1212;margin-bottom:4px;">Zone: ${escapeHtml(p.cell_id)}</div>
          <div style="display:flex;justify-content:space-between;margin:3px 0;">
            <span>Flood Probability:</span>
            <strong style="color:${color}">${probPct}% (${escapeHtml(p.risk_band)})</strong>
          </div>
          <div style="display:flex;justify-content:space-between;margin:3px 0;">
            <span>Observed 24h Rain:</span>
            <strong>${p.observed_rain_24h_mm.toFixed(1)} mm</strong>
          </div>
          <div style="display:flex;justify-content:space-between;margin:3px 0;">
            <span>Soil Moisture:</span>
            <strong>${p.soil_moisture.toFixed(2)} m³/m³</strong>
          </div>
          ${
            explanations
              ? `<div style="margin-top:6px;padding-top:6px;border-top:1px solid #e5e7eb;font-size:10px;color:#4b5563;">
                  <strong>Top Factors:</strong>${explanations}
                 </div>`
              : ""
          }
        </div>
      `;

      poly.bindPopup(popupHtml);
      poly.on("click", () => {
        if (onSelectCell) onSelectCell(feature);
      });
      group.addLayer(poly);
    });
  }, [hazardGrid, showHazard, onSelectCell]);

  // Update Hospitals
  useEffect(() => {
    const L = (window as LeafletWindow).L;
    const group = hospitalLayerRef.current;
    if (!L || !group) return;

    group.clearLayers();
    if (!showHospitals) return;

    hospitals.forEach((h) => {
      const icon = L.divIcon({
        className: "disaster-pulse-leaflet-marker",
        html: `<div style="width:24px;height:24px;border-radius:50%;background:#2563eb;color:#ffffff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,0.4);border:2px solid #ffffff;">H</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
        popupAnchor: [0, -12],
      });

      const marker = L.marker([h.latitude, h.longitude], { icon });
      const popupHtml = `
        <div style="font-family:Inter,sans-serif;font-size:12px;color:#111817;min-width:200px;">
          <div style="font-weight:700;font-size:13px;color:#1e3a8a;margin-bottom:4px;">${escapeHtml(h.name)}</div>
          <div style="font-size:10px;color:#6b7280;margin-bottom:6px;">${escapeHtml(h.district || "Tamil Nadu")} · ${escapeHtml(h.source)}</div>
          <div style="padding:6px;background:#f3f4f6;border-radius:4px;margin-bottom:6px;">
            <div style="font-size:11px;font-weight:600;color:#374151;">Available Beds: <span style="color:#6b7280;font-weight:normal;">Live telemetry unavailable</span></div>
            <div style="font-size:9px;color:#9ca3af;margin-top:2px;">(MoHFW/State portal publishes no live open bed API)</div>
          </div>
          ${h.distance_km !== undefined ? `<div style="font-size:11px;color:#374151;">Distance: <strong>${h.distance_km.toFixed(1)} km</strong></div>` : ""}
        </div>
      `;
      marker.bindPopup(popupHtml);
      group.addLayer(marker);
    });
  }, [hospitals, showHospitals]);

  // Update Shelters
  useEffect(() => {
    const L = (window as LeafletWindow).L;
    const group = shelterLayerRef.current;
    if (!L || !group) return;

    group.clearLayers();
    if (!showShelters) return;

    shelters.forEach((s) => {
      const isOfficial = s.facility_category === "official_cyclone_shelter";
      const icon = L.divIcon({
        className: "disaster-pulse-leaflet-marker",
        html: `<div style="width:24px;height:24px;border-radius:50%;background:#059669;color:#ffffff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,0.4);border:2px solid #ffffff;">S</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
        popupAnchor: [0, -12],
      });

      const marker = L.marker([s.latitude, s.longitude], { icon });
      const popupHtml = `
        <div style="font-family:Inter,sans-serif;font-size:12px;color:#111817;min-width:200px;">
          <div style="font-weight:700;font-size:13px;color:#065f46;margin-bottom:4px;">${escapeHtml(s.name)}</div>
          <div style="font-size:10px;color:#6b7280;margin-bottom:6px;">${escapeHtml(s.district || "Tamil Nadu")} · ${escapeHtml(s.source)}</div>
          <div style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600;background:${isOfficial ? "#d1fae5" : "#fef3c7"};color:${isOfficial ? "#065f46" : "#92400e"};margin-bottom:6px;">
            ${isOfficial ? "OFFICIAL SDMA SHELTER" : "CANDIDATE RELIEF FACILITY"}
          </div>
          ${s.capacity ? `<div style="font-size:11px;color:#374151;">Capacity: <strong>${s.capacity} persons</strong></div>` : ""}
          ${s.distance_km !== undefined ? `<div style="font-size:11px;color:#374151;">Distance: <strong>${s.distance_km.toFixed(1)} km</strong></div>` : ""}
        </div>
      `;
      marker.bindPopup(popupHtml);
      group.addLayer(marker);
    });
  }, [shelters, showShelters]);

  // Update Incidents
  useEffect(() => {
    const L = (window as LeafletWindow).L;
    const group = incidentLayerRef.current;
    if (!L || !group) return;

    group.clearLayers();
    if (!showIncidents) return;

    incidents.forEach((inc) => {
      const icon = L.divIcon({
        className: "disaster-pulse-leaflet-marker",
        html: `<div style="width:22px;height:22px;border-radius:50%;background:#ea580c;color:#ffffff;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,0.4);border:2px solid #ffffff;">!</div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11],
        popupAnchor: [0, -11],
      });

      const marker = L.marker([inc.latitude, inc.longitude], { icon });
      const popupHtml = `
        <div style="font-family:Inter,sans-serif;font-size:12px;color:#111817;min-width:210px;">
          <div style="display:inline-block;padding:2px 6px;border-radius:3px;font-size:9px;font-weight:700;background:#ffedd5;color:#c2410c;margin-bottom:6px;">
            UNVERIFIED CITIZEN REPORT
          </div>
          <div style="font-weight:700;font-size:13px;color:#9a3412;margin-bottom:4px;">Type: ${escapeHtml(inc.incident_type)}</div>
          <p style="font-size:11px;color:#374151;margin:4px 0 8px 0;">${escapeHtml(inc.description)}</p>
          <div style="font-size:10px;color:#6b7280;">Severity: <strong>${escapeHtml(inc.severity)}</strong> · By: ${escapeHtml(inc.reported_by)}</div>
        </div>
      `;
      marker.bindPopup(popupHtml);
      group.addLayer(marker);
    });
  }, [incidents, showIncidents]);

  // Update Routes
  useEffect(() => {
    const L = (window as LeafletWindow).L;
    const group = routeLayerRef.current;
    if (!L || !group) return;

    group.clearLayers();
    if (!showRoutes || !routes) return;

    // Draw Fastest Route (Blue dashed)
    if (routes.fastest && routes.fastest.feasible && routes.fastest.geometry?.coordinates) {
      const latlngs = routes.fastest.geometry.coordinates.map(([lon, lat]) => [lat, lon] as [number, number]);
      const fastestLine = L.polyline(latlngs, {
        color: "#60a5fa",
        weight: 4,
        dashArray: "6, 8",
        opacity: 0.8,
      });
      fastestLine.bindTooltip(
        `FASTEST: ${routes.fastest.total_distance_km.toFixed(1)} km · ${routes.fastest.risk_verdict}`,
        { sticky: true }
      );
      group.addLayer(fastestLine);
    }

    // Draw Safest Route (Teal solid with highlighted exposed segments)
    if (routes.safest && routes.safest.feasible && routes.safest.geometry?.coordinates) {
      const latlngs = routes.safest.geometry.coordinates.map(([lon, lat]) => [lat, lon] as [number, number]);
      const safestLine = L.polyline(latlngs, {
        color: "#14b8a6",
        weight: 6,
        opacity: 0.95,
      });
      safestLine.bindTooltip(
        `SAFEST: ${routes.safest.total_distance_km.toFixed(1)} km · ${routes.safest.risk_verdict}`,
        { sticky: true }
      );
      group.addLayer(safestLine);

      // Draw highlighted critical segments on safest route if any
      if (routes.safest.hazard_segments) {
        routes.safest.hazard_segments.forEach((seg) => {
          if (seg.risk_band === "critical" || seg.risk_band === "high") {
            const segLatLngs = seg.coordinates.map(([lon, lat]) => [lat, lon] as [number, number]);
            const hazardLine = L.polyline(segLatLngs, {
              color: seg.risk_band === "critical" ? "#ef4444" : "#f97316",
              weight: 8,
              opacity: 0.9,
            });
            hazardLine.bindTooltip(
              `WARNING: High Flood Exposure (${(seg.flood_probability * 100).toFixed(0)}%)`,
              { sticky: true }
            );
            group.addLayer(hazardLine);
          }
        });
      }

      // Auto-fit bounds to safest route
      if (mapInstanceRef.current && latlngs.length > 0) {
        mapInstanceRef.current.fitBounds(latlngs, { padding: [40, 40] });
      }
    }
  }, [routes, showRoutes]);

  // Update Active / Survivor Pin
  useEffect(() => {
    const L = (window as LeafletWindow).L;
    const group = pinLayerRef.current;
    if (!L || !group) return;

    group.clearLayers();
    if (!activePin) return;

    const icon = L.divIcon({
      className: "disaster-pulse-leaflet-marker",
      html: `<div style="width:28px;height:28px;border-radius:50%;background:#ef4444;color:#ffffff;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:14px;box-shadow:0 0 12px #ef4444;border:3px solid #ffffff;animation:pulse 1.5s infinite;">★</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
      popupAnchor: [0, -14],
    });

    const marker = L.marker(activePin, { icon });
    marker.bindPopup(
      `<div style="font-family:Inter,sans-serif;font-size:12px;color:#111817;">
        <strong>Selected Location / Survivor GPS</strong><br>
        Lat: ${activePin[0].toFixed(5)}, Lon: ${activePin[1].toFixed(5)}
      </div>`
    );
    group.addLayer(marker);
  }, [activePin]);

  // Focus on district change
  useEffect(() => {
    if (focusDistrictCoords && mapInstanceRef.current) {
      mapInstanceRef.current.setView(focusDistrictCoords, 10);
    }
  }, [focusDistrictCoords]);

  return (
    <div className="relative w-full h-[580px] bg-[#070909] border border-white/10 rounded-lg overflow-hidden">
      {/* Map Element */}
      <div ref={containerRef} className="w-full h-full" />

      {/* Layer Controls Bar */}
      <div className="absolute top-3 right-3 z-[400] flex flex-wrap gap-2 bg-[#0d1212]/90 backdrop-blur-md p-2 rounded border border-white/10 text-xs">
        <label className="flex items-center gap-1.5 text-gray-300 cursor-pointer hover:text-white">
          <input
            type="checkbox"
            checked={showHazard}
            onChange={(e) => setShowHazard(e.target.checked)}
            className="accent-[#72cfc5]"
          />
          Hazard Grid
        </label>
        <label className="flex items-center gap-1.5 text-gray-300 cursor-pointer hover:text-white">
          <input
            type="checkbox"
            checked={showHospitals}
            onChange={(e) => setShowHospitals(e.target.checked)}
            className="accent-[#2563eb]"
          />
          Hospitals
        </label>
        <label className="flex items-center gap-1.5 text-gray-300 cursor-pointer hover:text-white">
          <input
            type="checkbox"
            checked={showShelters}
            onChange={(e) => setShowShelters(e.target.checked)}
            className="accent-[#059669]"
          />
          Shelters
        </label>
        <label className="flex items-center gap-1.5 text-gray-300 cursor-pointer hover:text-white">
          <input
            type="checkbox"
            checked={showIncidents}
            onChange={(e) => setShowIncidents(e.target.checked)}
            className="accent-[#ea580c]"
          />
          Citizen Reports
        </label>
        {routes && (
          <label className="flex items-center gap-1.5 text-gray-300 cursor-pointer hover:text-white">
            <input
              type="checkbox"
              checked={showRoutes}
              onChange={(e) => setShowRoutes(e.target.checked)}
              className="accent-[#14b8a6]"
            />
            Routes
          </label>
        )}
      </div>

      {/* Map Legend */}
      <div className="absolute bottom-4 left-4 z-[400] bg-[#0d1212]/90 backdrop-blur-md p-3 rounded border border-white/10 text-[11px] space-y-1.5 pointer-events-auto">
        <div className="text-[9px] font-bold tracking-wider text-gray-400 uppercase mb-1">Hazard Risk Band</div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-[#ef4444] opacity-75 inline-block" />
          <span className="text-gray-300">Critical (&gt;80%)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-[#f97316] opacity-75 inline-block" />
          <span className="text-gray-300">High (60–80%)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-[#f59e0b] opacity-75 inline-block" />
          <span className="text-gray-300">Moderate (35–60%)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-[#10b981] opacity-75 inline-block" />
          <span className="text-gray-300">Low (&lt;35%)</span>
        </div>
        <div className="pt-2 mt-2 border-t border-white/10 text-[10px] text-gray-400">
          <div><strong className="text-blue-400">H</strong> Hospital (Live beds null)</div>
          <div><strong className="text-emerald-400">S</strong> Evacuation Shelter</div>
          <div><strong className="text-orange-400">!</strong> Unverified Citizen Report</div>
        </div>
      </div>
    </div>
  );
}
