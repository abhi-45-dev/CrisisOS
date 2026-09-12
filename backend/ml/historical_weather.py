from __future__ import annotations

"""Historical Meteorological Feature Provider for FloodNow TN.

Connects to Open-Meteo Historical Archive API (ERA5 / ERA5-Land reanalysis),
caches raw hourly series to disk with full provenance manifests, and extracts
strict pre-event antecedent precipitation accumulations and root-zone soil moisture.

ANTI-LEAKAGE INVARIANT:
The function get_environmental_features(lat, lon, timestamp) does NOT accept any target,
class label, or positive/negative indicator. Exactly the same function is used for both
positive recorded floods and weak-negative observations.
"""

import hashlib
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

CACHE_DIR = Path(__file__).resolve().parent / 'cache' / 'historical_weather'
OPEN_METEO_ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'


class HistoricalWeatherClient:
    """Retrieves and caches historical ERA5/ERA5-Land reanalysis weather from Open-Meteo."""

    def __init__(self, cache_dir: Path | None = None, request_pause_seconds: float = 0.05):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_pause = request_pause_seconds

    def _cache_key(self, lat: float, lon: float, start_date: str, end_date: str) -> Path:
        key_str = f'{lat:.4f}_{lon:.4f}_{start_date}_{end_date}'
        hash_digest = hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:16]
        return self.cache_dir / f'weather_{round(lat, 2)}_{round(lon, 2)}_{start_date}_{end_date}_{hash_digest}.json'

    def fetch_window(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        max_retries: int = 4,
    ) -> dict[str, Any]:
        """Fetch hourly historical precipitation and soil moisture for coordinate and date range.

        Uses local disk cache if present. On miss, queries Open-Meteo with retries and exponential backoff.
        NEVER substitutes synthetic weather on failure.
        """
        cache_file = self._cache_key(lat, lon, start_date, end_date)
        manifest_file = cache_file.with_suffix('.manifest.json')

        if cache_file.exists() and manifest_file.exists():
            try:
                with cache_file.open('r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        params = {
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'precipitation,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm',
            'timezone': 'Asia/Kolkata',
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                if self.request_pause > 0:
                    time.sleep(self.request_pause)

                with httpx.Client(timeout=25.0) as client:
                    resp = client.get(OPEN_METEO_ARCHIVE_URL, params=params)
                    if resp.status_code == 429:
                        # Rate limited, back off longer
                        backoff = 2.0 * (2 ** attempt)
                        time.sleep(backoff)
                        continue

                    resp.raise_for_status()
                    data = resp.json()

                    # Validate structure
                    hourly = data.get('hourly', {})
                    if 'time' not in hourly or 'precipitation' not in hourly:
                        raise ValueError(f'Incomplete Open-Meteo payload for ({lat}, {lon})')

                    # Write cache + manifest
                    with cache_file.open('w', encoding='utf-8') as f:
                        json.dump(data, f)

                    manifest = {
                        'provider': 'Open-Meteo Historical Archive (ERA5 / ERA5-Land)',
                        'url': OPEN_METEO_ARCHIVE_URL,
                        'retrieved_at': datetime.now(timezone.utc).isoformat(),
                        'latitude': round(lat, 4),
                        'longitude': round(lon, 4),
                        'start_date': start_date,
                        'end_date': end_date,
                        'params': params,
                        'hourly_count': len(hourly.get('time', [])),
                        'status': 'CACHED',
                    }
                    with manifest_file.open('w', encoding='utf-8') as f:
                        json.dump(manifest, f, indent=2)

                    return data

            except Exception as e:
                last_error = e
                backoff = 1.0 * (2 ** attempt)
                time.sleep(backoff)

        raise RuntimeError(
            f'Failed to retrieve genuine historical weather for ({lat}, {lon}) '
            f'over [{start_date} to {end_date}]: {last_error}'
        )


_default_client = HistoricalWeatherClient()


def get_environmental_features(
    latitude: float,
    longitude: float,
    prediction_timestamp: datetime | str,
    client: HistoricalWeatherClient | None = None,
) -> dict[str, Any]:
    """Extract strictly pre-event antecedent environmental features for coordinate and timestamp.

    ANTI-LEAKAGE INVARIANT:
    Does NOT receive target, class, or positive/negative indicator.
    Accepts ONLY location and prediction timestamp.

    Pre-event time semantics:
    If a date string 'YYYY-MM-DD' or midnight timestamp is given, the antecedent evaluation window
    ends at 23:00 IST on the previous day. This guarantees no rainfall occurring during or after the
    flood event is leaked into antecedent predictors.
    """
    cli = client or _default_client

    if isinstance(prediction_timestamp, str):
        clean_ts = prediction_timestamp.strip()
        if len(clean_ts) == 10:
            dt = datetime.strptime(clean_ts, '%Y-%m-%d')
            cutoff = dt - timedelta(hours=1)
        else:
            dt = datetime.fromisoformat(clean_ts.replace('Z', '+00:00'))
            if dt.hour == 0 and dt.minute == 0:
                cutoff = dt - timedelta(hours=1)
            else:
                cutoff = dt
    else:
        dt = prediction_timestamp
        if dt.hour == 0 and dt.minute == 0:
            cutoff = dt - timedelta(hours=1)
        else:
            cutoff = dt

    month = cutoff.month

    # Antecedent 72 hours requires 4 calendar days preceding cutoff
    start_date = (cutoff - timedelta(hours=72)).strftime('%Y-%m-%d')
    end_date = cutoff.strftime('%Y-%m-%d')

    raw_data = cli.fetch_window(latitude, longitude, start_date, end_date)
    hourly = raw_data.get('hourly', {})
    times = hourly.get('time', [])
    precip_raw = hourly.get('precipitation', [])
    sm0_raw = hourly.get('soil_moisture_0_to_7cm', [])
    sm7_raw = hourly.get('soil_moisture_7_to_28cm', [])

    cutoff_str = cutoff.strftime('%Y-%m-%dT%H:%M')
    valid_indices = [i for i, t in enumerate(times) if t <= cutoff_str]

    if not valid_indices:
        raise ValueError(f'No valid hourly observations prior to cutoff {cutoff_str}')

    precip = [precip_raw[i] for i in valid_indices]
    sm0 = [sm0_raw[i] for i in valid_indices]
    sm7 = [sm7_raw[i] for i in valid_indices]

    def _sum_trailing(seq: list[Any], window_h: int) -> float | None:
        if len(seq) < window_h:
            return None
        sub = seq[-window_h:]
        if any(x is None for x in sub):
            return None
        return round(float(sum(sub)), 2)

    def _latest_val(seq: list[Any]) -> float | None:
        for val in reversed(seq):
            if val is not None:
                return round(float(val), 4)
        return None

    r_1h = _sum_trailing(precip, 1)
    r_3h = _sum_trailing(precip, 3)
    r_6h = _sum_trailing(precip, 6)
    r_24h = _sum_trailing(precip, 24)
    r_72h = _sum_trailing(precip, 72)

    latest_sm0 = _latest_val(sm0)
    latest_sm7 = _latest_val(sm7)

    return {
        'latitude': round(float(latitude), 4),
        'longitude': round(float(longitude), 4),
        'month_sin': round(math.sin(2 * math.pi * month / 12.0), 4),
        'month_cos': round(math.cos(2 * math.pi * month / 12.0), 4),
        'rain_1h_mm': r_1h,
        'rain_3h_mm': r_3h,
        'rain_6h_mm': r_6h,
        'rain_24h_mm': r_24h,
        'rain_72h_mm': r_72h,
        'soil_moisture_0_7cm': latest_sm0,
        'soil_moisture_7_28cm': latest_sm7,
    }
