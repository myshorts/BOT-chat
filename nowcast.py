import argparse
import datetime as dt
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import requests
from PIL import Image, ImageDraw

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
STATE_FILE = Path(".nowcast_state.json")

# Tọa độ Sân bay Phù Cát & vùng không phận 15km (kèm tham chiếu KV 50 - KV 58)
STATION = {
    "code": "VVPC",
    "name": "Sân bay Phù Cát (Trung đoàn 925)",
    "lat": 13.9550,
    "lon": 109.0422,
    "airspace_radius_km": 15.0,
}

# Tọa độ tham chiếu các khu vực KV 50 - KV 58 quanh Phù Cát (15 km)
KV_ZONES: Dict[str, Tuple[float, float]] = {
    "KV 50": (13.9550, 109.0422),
    "KV 51": (14.0150, 109.0100),
    "KV 52": (14.0050, 109.0950),
    "KV 53": (13.9550, 109.1350),
    "KV 54": (13.8950, 109.1050),
    "KV 55": (13.8850, 109.0250),
    "KV 56": (13.9150, 108.9500),
    "KV 57": (13.9850, 108.9550),
    "KV 58": (13.9550, 108.9150),
}

# Ngưỡng cảnh báo (chỉnh tại dict TH đầu file)
TH: Dict[str, Any] = {
    "storm_dbz": 45.0,
    "storm_radius_km": 30.0,
    "storm_ct_c": -55.0,
    "approach_window_min": 45,
    "shower_window_min": 30,
    "shower_dbz_min": 20.0,
    "near_station_radius_km": 15.0,
    "rapid_conv_dbz_30m": 10.0,
    "rapid_cool_c_30m": 8.0,
    "cooldown_minutes": {
        5: 15,
        4: 20,
        3: 25,
        2: 30,
        1: 45,
        0: 55,
    },
}

# Hiệu chuẩn IR (CAL_WARM / CAL_COLD) giữ như trang web hiện tại
CAL_WARM: Dict[str, Any] = {
    "pixel": 235.0,
    "temp_c": 25.0,
}
CAL_COLD: Dict[str, Any] = {
    "pixel": 25.0,
    "temp_c": -80.0,
}

# Bảng màu Universal Blue (RGB -> dBZ gần đúng)
UNIVERSAL_BLUE_PALETTE: List[Tuple[Tuple[int, int, int], float]] = [
    ((4, 233, 231), 5.0),
    ((1, 159, 244), 10.0),
    ((3, 0, 244), 15.0),
    ((2, 253, 2), 20.0),
    ((1, 197, 1), 25.0),
    ((0, 142, 0), 30.0),
    ((253, 248, 2), 35.0),
    ((229, 188, 0), 40.0),
    ((233, 122, 0), 45.0),
    ((244, 0, 0), 50.0),
    ((212, 0, 0), 55.0),
    ((156, 0, 0), 60.0),
    ((255, 255, 255), 65.0),
]

RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps.json"
OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"
AVIATION_WEATHER_METAR_API = "https://aviationweather.gov/api/data/metar"


def now_vn() -> dt.datetime:
    return dt.datetime.now(TZ_VN)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def bearing_to_vi_direction(deg: float) -> str:
    dirs = [
        "Bắc", "Đông Bắc", "Đông", "Đông Nam",
        "Nam", "Tây Nam", "Tây", "Tây Bắc",
    ]
    idx = int(((deg + 22.5) % 360.0) // 45.0)
    return dirs[idx]


def offset_latlon(lat: float, lon: float, dist_km: float, brg_deg: float) -> Tuple[float, float]:
    r = 6371.0
    ang = dist_km / r
    brg = math.radians(brg_deg)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(ang) + math.cos(p1) * math.sin(ang) * math.cos(brg))
    l2 = l1 + math.atan2(
        math.sin(brg) * math.sin(ang) * math.cos(p1),
        math.cos(ang) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), ((math.degrees(l2) + 180.0) % 360.0) - 180.0


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    lat_c = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat_c)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def rgb_to_dbz_array(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    dbz = np.full((h, w), np.nan, dtype=np.float32)
    valid = (rgb[:, :, 0] > 10) | (rgb[:, :, 1] > 10) | (rgb[:, :, 2] > 10)
    if not np.any(valid):
        return dbz

    palette_rgb = np.array([c for c, _ in UNIVERSAL_BLUE_PALETTE], dtype=np.float32)
    palette_val = np.array([v for _, v in UNIVERSAL_BLUE_PALETTE], dtype=np.float32)

    pts = rgb[valid].astype(np.float32)
    dists = np.linalg.norm(pts[:, None, :] - palette_rgb[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    min_d = dists[np.arange(len(pts)), nearest]

    mapped = np.where(min_d <= 65.0, palette_val[nearest], np.nan)
    dbz[valid] = mapped
    return dbz


def dbz_to_rain_mm_hr(dbz: float) -> float:
    if np.isnan(dbz) or dbz < 15.0:
        return 0.0
    z = 10.0 ** (dbz / 10.0)
    return round((z / 200.0) ** (1.0 / 1.6), 1)


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_sent": {}, "history": []}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_cell_grid(
    dbz_grid: np.ndarray,
    ct_grid: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> Dict[str, Any]:
    dist_km = haversine_km(STATION["lat"], STATION["lon"], grid_lat, grid_lon)
    brg_deg = bearing_deg(STATION["lat"], STATION["lon"], grid_lat, grid_lon)

    valid_dbz = ~np.isnan(dbz_grid)
    if not np.any(valid_dbz):
        return {
            "max_dbz_30km": 0.0,
            "max_dbz_15km": 0.0,
            "station_dbz": 0.0,
            "min_ct_30km": 20.0,
            "min_ct_15km": 20.0,
            "strongest_cell": None,
            "nearest_rain_cell": None,
        }

    near_st_mask = valid_dbz & (dist_km <= 4.0)
    station_dbz = float(np.nanmax(dbz_grid[near_st_mask])) if np.any(near_st_mask) else 0.0

    m15 = valid_dbz & (dist_km <= TH["near_station_radius_km"])
    m30 = valid_dbz & (dist_km <= TH["storm_radius_km"])

    max_dbz_15km = float(np.nanmax(dbz_grid[m15])) if np.any(m15) else 0.0
    max_dbz_30km = float(np.nanmax(dbz_grid[m30])) if np.any(m30) else 0.0

    valid_ct15 = m15 & (~np.isnan(ct_grid))
    valid_ct30 = m30 & (~np.isnan(ct_grid))
    min_ct_15km = float(np.nanmin(ct_grid[valid_ct15])) if np.any(valid_ct15) else 15.0
    min_ct_30km = float(np.nanmin(ct_grid[valid_ct30])) if np.any(valid_ct30) else 15.0

    strongest_cell = None
    if np.any(m30):
        sub_dbz = np.where(m30, dbz_grid, -999.0)
        idx = np.unravel_index(np.nanargmax(sub_dbz), sub_dbz.shape)
        c_lat = float(grid_lat[idx])
        c_lon = float(grid_lon[idx])
        c_dist = float(dist_km[idx])
        c_brg = float(brg_deg[idx])
        c_dbz = float(dbz_grid[idx])
        c_ct = float(ct_grid[idx]) if not np.isnan(ct_grid[idx]) else 15.0
        strongest_cell = {
            "lat": c_lat,
            "lon": c_lon,
            "dist_km": round(c_dist, 1),
            "bearing_deg": round(c_brg, 1),
            "direction": bearing_to_vi_direction(c_brg),
            "dbz": round(c_dbz, 1),
            "cloud_top_c": round(c_ct, 1),
        }

    nearest_rain_cell = None
    rain_mask = valid_dbz & (dist_km <= TH["storm_radius_km"]) & (dbz_grid >= TH["shower_dbz_min"])
    if np.any(rain_mask):
        sub_dist = np.where(rain_mask, dist_km, 9999.0)
        idx = np.unravel_index(np.nanargmin(sub_dist), sub_dist.shape)
        r_lat = float(grid_lat[idx])
        r_lon = float(grid_lon[idx])
        r_dist = float(dist_km[idx])
        r_brg = float(brg_deg[idx])
        r_dbz = float(dbz_grid[idx])
        nearest_rain_cell = {
            "lat": r_lat,
            "lon": r_lon,
            "dist_km": round(r_dist, 1),
            "bearing_deg": round(r_brg, 1),
            "direction": bearing_to_vi_direction(r_brg),
            "dbz": round(r_dbz, 1),
        }

    return {
        "max_dbz_30km": round(max_dbz_30km, 1),
        "max_dbz_15km": round(max_dbz_15km, 1),
        "station_dbz": round(station_dbz, 1),
        "min_ct_30km": round(min_ct_30km, 1),
        "min_ct_15km": round(min_ct_15km, 1),
        "strongest_cell": strongest_cell,
        "nearest_rain_cell": nearest_rain_cell,
    }


def extrapolate_motion(
    prev_cell: Optional[Dict[str, Any]],
    curr_cell: Optional[Dict[str, Any]],
    dt_minutes: float,
) -> Optional[Dict[str, Any]]:
    if not prev_cell or not curr_cell or dt_minutes <= 0:
        return None

    lat1, lon1 = prev_cell["lat"], prev_cell["lon"]
    lat2, lon2 = curr_cell["lat"], curr_cell["lon"]
    moved_km = haversine_km(lat1, lon1, lat2, lon2)
    if moved_km < 0.5:
        return None

    move_brg = bearing_deg(lat1, lon1, lat2, lon2)
    speed_kmh = moved_km / (dt_minutes / 60.0)

    to_station_brg = bearing_deg(lat2, lon2, STATION["lat"], STATION["lon"])
    angle_diff = abs(((move_brg - to_station_brg + 180.0) % 360.0) - 180.0)
    dist_to_station = haversine_km(lat2, lon2, STATION["lat"], STATION["lon"])

    eta_min: Optional[float] = None
    if angle_diff <= 45.0 and speed_kmh > 3.0:
        eta_min = (dist_to_station / speed_kmh) * 60.0

    return {
        "move_bearing_deg": round(move_brg, 1),
        "move_direction": bearing_to_vi_direction(move_brg),
        "speed_kmh": round(speed_kmh, 1),
        "eta_min": round(eta_min, 0) if eta_min is not None else None,
        "dist_to_station_km": round(dist_to_station, 1),
    }


def fetch_real_radar_and_ir() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = 128
    half_span_deg = 0.45
    lats = np.linspace(STATION["lat"] + half_span_deg, STATION["lat"] - half_span_deg, size)
    lons = np.linspace(STATION["lon"] - half_span_deg, STATION["lon"] + half_span_deg, size)
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    dbz_grid = np.full((size, size), np.nan, dtype=np.float32)
    ct_grid = np.full((size, size), np.nan, dtype=np.float32)

    rv_resp = requests.get(RAINVIEWER_API, timeout=20)
    rv_resp.raise_for_status()
    rv_data = rv_resp.json()
    host = rv_data.get("host", "https://tilecache.rainviewer.com")
    past_frames = rv_data.get("radar", {}).get("past", [])
    if past_frames:
        latest_path = past_frames[-1]["path"]
        zoom = 7
        tx, ty = latlon_to_tile_xy(STATION["lat"], STATION["lon"], zoom)
        tile_url = f"{host}{latest_path}/256/{zoom}/{int(tx)}/{int(ty)}/1/1_1.png"
        t_resp = requests.get(tile_url, timeout=20)
        t_resp.raise_for_status()
        img = Image.open(BytesIO(t_resp.content)).convert("RGB")
        fx = (tx - int(tx)) * 256.0
        fy = (ty - int(ty)) * 256.0
        crop_half = 48
        x0 = max(0, int(fx - crop_half))
        y0 = max(0, int(fy - crop_half))
        x1 = min(256, int(fx + crop_half))
        y1 = min(256, int(fy + crop_half))
        cropped = img.crop((x0, y0, x1, y1)).resize((size, size), Image.Resampling.BILINEAR)
        dbz_grid = rgb_to_dbz_array(np.asarray(cropped))

    ir_url = os.environ.get("JMA_IR_URL", "").strip()
    if ir_url:
        ir_resp = requests.get(ir_url, timeout=20)
        ir_resp.raise_for_status()
        ir_img = Image.open(BytesIO(ir_resp.content)).convert("L")
        ir_resized = ir_img.resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(ir_resized, dtype=np.float32)
        p_warm = float(CAL_WARM["pixel"])
        t_warm = float(CAL_WARM["temp_c"])
        p_cold = float(CAL_COLD["pixel"])
        t_cold = float(CAL_COLD["temp_c"])
        ratio = np.clip((arr - p_cold) / max(1e-6, (p_warm - p_cold)), 0.0, 1.0)
        ct_grid = (t_cold + ratio * (t_warm - t_cold)).astype(np.float32)

    return dbz_grid, ct_grid, grid_lat, grid_lon


def build_demo_frames() -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    size = 128
    half_span_deg = 0.45
    lats = np.linspace(STATION["lat"] + half_span_deg, STATION["lat"] - half_span_deg, size)
    lons = np.linspace(STATION["lon"] - half_span_deg, STATION["lon"] + half_span_deg, size)
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    c1_lat, c1_lon = offset_latlon(STATION["lat"], STATION["lon"], 35.0, 315.0)
    c2_lat, c2_lon = offset_latlon(c1_lat, c1_lon, 10.0, 90.0)

    frames = []
    for clat, clon, peak_dbz, peak_ct, mins_ago in [
        (c1_lat, c1_lon, 44.0, -52.0, 20.0),
        (c2_lat, c2_lon, 52.0, -61.0, 0.0),
    ]:
        png_img = Image.new("RGB", (size, size), (0, 0, 0))
        draw = ImageDraw.Draw(png_img)
        px = int(((clon - (STATION["lon"] - half_span_deg)) / (2 * half_span_deg)) * size)
        py = int((((STATION["lat"] + half_span_deg) - clat) / (2 * half_span_deg)) * size)
        color = (212, 0, 0) if peak_dbz >= 50 else (233, 122, 0)
        r_px = 7
        draw.ellipse((px - r_px, py - r_px, px + r_px, py + r_px), fill=color)

        gray_val = int(
            np.interp(
                peak_ct,
                [float(CAL_COLD["temp_c"]), float(CAL_WARM["temp_c"])],
                [float(CAL_COLD["pixel"]), float(CAL_WARM["pixel"])],
            )
        )
        ir_img = Image.new("L", (size, size), int(CAL_WARM["pixel"]))
        ir_draw = ImageDraw.Draw(ir_img)
        ir_draw.ellipse((px - r_px - 2, py - r_px - 2, px + r_px + 2, py + r_px + 2), fill=gray_val)
        jpeg_buf = BytesIO()
        ir_img.save(jpeg_buf, format="JPEG", quality=92)
        jpeg_buf.seek(0)
        decoded_ir = np.asarray(Image.open(jpeg_buf).convert("L"), dtype=np.float32)

        p_warm, t_warm = float(CAL_WARM["pixel"]), float(CAL_WARM["temp_c"])
        p_cold, t_cold = float(CAL_COLD["pixel"]), float(CAL_COLD["temp_c"])
        ratio = np.clip((decoded_ir - p_cold) / max(1e-6, (p_warm - p_cold)), 0.0, 1.0)
        ct_grid = (t_cold + ratio * (t_warm - t_cold)).astype(np.float32)
        dbz_grid = rgb_to_dbz_array(np.asarray(png_img))

        frames.append((dbz_grid, ct_grid, grid_lat, grid_lon, mins_ago))

    return frames


def fetch_station_weather_and_metar() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "temp_c": None,
        "feels_like_c": None,
        "humidity": None,
        "next_3h_summary": "Chưa có dữ liệu dự báo 3 giờ tới",
        "day_minmax": None,
        "rain_hours": [],
        "metar": "METAR VVPC chưa cập nhật",
    }

    try:
        params = {
            "latitude": STATION["lat"],
            "longitude": STATION["lon"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m",
            "hourly": "temperature_2m,precipitation_probability,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Ho_Chi_Minh",
            "forecast_days": 1,
        }
        r = requests.get(OPEN_METEO_API, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        curr = data.get("current", {})
        info["temp_c"] = curr.get("temperature_2m")
        info["feels_like_c"] = curr.get("apparent_temperature")
        info["humidity"] = curr.get("relative_humidity_2m")

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        probs = hourly.get("precipitation_probability", [])
        temps = hourly.get("temperature_2m", [])

        now_str = now_vn().strftime("%Y-%m-%dT%H:00")
        start_idx = 0
        for i, t_str in enumerate(times):
            if t_str >= now_str:
                start_idx = i
                break

        next3_parts = []
        for j in range(start_idx, min(start_idx + 3, len(times))):
            hh = times[j].split("T")[1][:5]
            t_val = temps[j] if j < len(temps) else "?"
            p_val = probs[j] if j < len(probs) else 0
            next3_parts.append(f"{hh}: {t_val}°C, mưa {p_val}%")
        if next3_parts:
            info["next_3h_summary"] = " | ".join(next3_parts)

        rain_hours = []
        for j, t_str in enumerate(times):
            p_val = probs[j] if j < len(probs) else 0
            if p_val is not None and p_val >= 40:
                rain_hours.append(f"{t_str.split('T')[1][:5]} ({p_val}%)")
        info["rain_hours"] = rain_hours

        daily = data.get("daily", {})
        d_max = (daily.get("temperature_2m_max") or [None])[0]
        d_min = (daily.get("temperature_2m_min") or [None])[0]
        if d_min is not None and d_max is not None:
            info["day_minmax"] = (d_min, d_max)
    except Exception:
        pass

    try:
        m_resp = requests.get(AVIATION_WEATHER_METAR_API, params={"ids": STATION["code"], "format": "raw"}, timeout=12)
        if m_resp.ok and m_resp.text.strip():
            info["metar"] = m_resp.text.strip().splitlines()[0]
    except Exception:
        pass

    return info


def evaluate_alert(
    curr_stats: Dict[str, Any],
    prev_stats: Optional[Dict[str, Any]],
    motion: Optional[Dict[str, Any]],
    weather: Dict[str, Any],
    now_dt: dt.datetime,
) -> Optional[Dict[str, Any]]:
    strongest = curr_stats.get("strongest_cell")
    nearest_rain = curr_stats.get("nearest_rain_cell")
    max_dbz_30 = curr_stats.get("max_dbz_30km", 0.0)
    min_ct_30 = curr_stats.get("min_ct_30km", 20.0)
    station_dbz = curr_stats.get("station_dbz", 0.0)

    if max_dbz_30 >= TH["storm_dbz"] and min_ct_30 <= TH["storm_ct_c"] and strongest:
        return {
            "level": 5,
            "key": "storm_active",
            "ntfy_priority": "urgent",
            "title": f"DÔNG ĐANG XẢY RA — {STATION['name']}",
            "body": (
                f"Echo {strongest['dbz']} dBZ, đỉnh mây {strongest['cloud_top_c']}°C, "
                f"cách {strongest['dist_km']} km hướng {strongest['direction']}."
            ),
        }

    if (
        strongest
        and strongest["dbz"] >= TH["storm_dbz"]
        and motion
        and motion.get("eta_min") is not None
        and 0 < motion["eta_min"] <= TH["approach_window_min"]
    ):
        eta = int(motion["eta_min"])
        return {
            "level": 4,
            "key": "storm_approaching",
            "ntfy_priority": "high",
            "title": f"DÔNG SẮP TỚI (~{eta} PHÚT) — {STATION['code']}",
            "body": (
                f"Ô echo {strongest['dbz']} dBZ (cách {strongest['dist_km']} km hướng {strongest['direction']}), "
                f"di chuyển hướng {motion['move_direction']} ~{motion['speed_kmh']} km/h, dự kiến tới sau ~{eta} phút."
            ),
        }

    if prev_stats and prev_stats.get("strongest_cell") and strongest:
        prev_c = prev_stats["strongest_cell"]
        dbz_gain = strongest["dbz"] - prev_c.get("dbz", strongest["dbz"])
        ct_cooling = prev_c.get("cloud_top_c", strongest["cloud_top_c"]) - strongest["cloud_top_c"]
        if dbz_gain >= TH["rapid_conv_dbz_30m"] or ct_cooling >= TH["rapid_cool_c_30m"]:
            return {
                "level": 3,
                "key": "rapid_convection",
                "ntfy_priority": "high",
                "title": f"ĐỐI LƯU PHÁT TRIỂN NHANH — {STATION['code']}",
                "body": (
                    f"Echo {strongest['dbz']} dBZ (tăng {dbz_gain:+.1f} dBZ), đỉnh mây {strongest['cloud_top_c']}°C "
                    f"(lạnh đi {ct_cooling:+.1f}°C), cách {strongest['dist_km']} km hướng {strongest['direction']}."
                ),
            }

    if nearest_rain:
        near_station = nearest_rain["dist_km"] <= TH["near_station_radius_km"]
        eta_rain = int(motion["eta_min"]) if (motion and motion.get("eta_min") is not None) else None
        approaching_30m = eta_rain is not None and 0 < eta_rain <= TH["shower_window_min"]
        if near_station or approaching_30m:
            eta_txt = f"tới sau ~{eta_rain} phút" if eta_rain else f"cách {nearest_rain['dist_km']} km hướng {nearest_rain['direction']}"
            return {
                "level": 2,
                "key": "shower_alert",
                "ntfy_priority": "default",
                "title": f"MƯA RÀO BẤT CHỢT — {STATION['code']}",
                "body": f"Echo {nearest_rain['dbz']} dBZ ({eta_txt}) trong vùng giám sát {STATION['name']}.",
            }

    if station_dbz >= 15.0:
        mm_hr = dbz_to_rain_mm_hr(station_dbz)
        return {
            "level": 1,
            "key": "rain_at_station",
            "ntfy_priority": "low",
            "title": f"ĐANG MƯA TẠI {STATION['code']}",
            "body": f"Phản hồi radar tại trạm ~{station_dbz} dBZ, cường độ mưa ước tính ~{mm_hr} mm/giờ.",
        }

    if 5 <= now_dt.hour <= 22:
        radar_line = (
            f"Radar: echo cao nhất 30 km là {max_dbz_30} dBZ; IR đỉnh mây thấp nhất {min_ct_30}°C."
            if max_dbz_30 > 0
            else f"Radar/IR: không có echo mưa đáng kể trong 30 km; đỉnh mây thấp nhất {min_ct_30}°C."
        )
        lines = [
            f"Nhiệt độ: {weather.get('temp_c', '?')}°C | Cảm giác: {weather.get('feels_like_c', '?')}°C | Độ ẩm: {weather.get('humidity', '?')}%",
            f"3 giờ tới: {weather.get('next_3h_summary')}",
            f"METAR: {weather.get('metar')}",
            radar_line,
        ]
        if now_dt.hour == 6:
            d_mm = weather.get("day_minmax")
            mm_txt = f"{d_mm[0]}°C – {d_mm[1]}°C" if d_mm else "N/A"
            r_hrs = weather.get("rain_hours") or []
            r_txt = ", ".join(r_hrs) if r_hrs else "ít khả năng mưa"
            lines.append(f"Cả ngày: {mm_txt} | Giờ có khả năng mưa: {r_txt}")

        return {
            "level": 0,
            "key": f"hourly_{now_dt.strftime('%Y%m%d_%H')}",
            "ntfy_priority": "low",
            "title": f"BẢN TIN {now_dt.strftime('%H:%M')} — {STATION['name']}",
            "body": "\n".join(lines),
        }

    return None


def can_send_alert(alert: Dict[str, Any], state: Dict[str, Any], now_dt: dt.datetime) -> bool:
    level = alert["level"]
    key = alert["key"]
    last_sent = state.get("last_sent", {})
    prev_iso = last_sent.get(key)
    if not prev_iso:
        return True
    try:
        prev_dt = dt.datetime.fromisoformat(prev_iso)
    except Exception:
        return True
    cooldown_min = TH["cooldown_minutes"].get(level, 20)
    elapsed_min = (now_dt - prev_dt).total_seconds() / 60.0
    return elapsed_min >= cooldown_min


def send_ntfy(alert: Dict[str, Any]) -> None:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        raise RuntimeError("Chưa cấu hình biến môi trường NTFY_TOPIC.")
    url = f"https://ntfy.sh/{topic.rstrip('/')}"
    tags = "rotating_light,thunder_cloud_and_rain" if alert["level"] == 5 else "partly_sunny"
    headers = {
        "Title": alert["title"].encode("utf-8"),
        "Priority": alert["ntfy_priority"],
        "Tags": tags,
    }
    resp = requests.post(url, data=alert["body"].encode("utf-8"), headers=headers, timeout=15)
    resp.raise_for_status()
    print(f"[NTFY SENT] Mức {alert['level']} -> {alert['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phù Cát Weather Nowcast (Radar + IR + METAR + ntfy)")
    parser.add_argument("--demo", action="store_true", help="Chạy dữ liệu giả lập, in kết quả")
    parser.add_argument("--dry", action="store_true", help="Chạy dữ liệu thật, chỉ in, không gửi")
    parser.add_argument("--test", action="store_true", help="Gửi 1 tin thử qua ntfy")
    args = parser.parse_args()

    now_dt = now_vn()

    if args.test:
        sample = {
            "level": 5,
            "key": "test_alert",
            "ntfy_priority": "urgent",
            "title": f"TEST CẢNH BÁO DÔNG — {STATION['name']}",
            "body": "Tin nhắn kiểm thử kênh ntfy cho iPhone (mức 5 nổi bật).",
        }
        send_ntfy(sample)
        return

    state = load_state()
    prev_stats: Optional[Dict[str, Any]] = None
    motion: Optional[Dict[str, Any]] = None

    if args.demo:
        demo_frames = build_demo_frames()
        f1_dbz, f1_ct, f1_lat, f1_lon, _ = demo_frames[0]
        f2_dbz, f2_ct, f2_lat, f2_lon, _ = demo_frames[1]
        prev_stats = analyze_cell_grid(f1_dbz, f1_ct, f1_lat, f1_lon)
        curr_stats = analyze_cell_grid(f2_dbz, f2_ct, f2_lat, f2_lon)
        motion = extrapolate_motion(prev_stats.get("strongest_cell"), curr_stats.get("strongest_cell"), dt_minutes=20.0)
        weather = {
            "temp_c": 27.5,
            "feels_like_c": 30.0,
            "humidity": 84,
            "next_3h_summary": "12:00: 28°C, mưa 60% | 13:00: 27°C, mưa 75% | 14:00: 26°C, mưa 50%",
            "day_minmax": (24.0, 31.5),
            "rain_hours": ["12:00 (60%)", "13:00 (75%)", "14:00 (50%)"],
            "metar": "METAR VVPC 190400Z 11008KT 8000 -TSRA FEW018 SCT030CB 27/24 Q1009",
        }
        print(f"[DEMO] Prev stats: {prev_stats}")
        print(f"[DEMO] Curr stats: {curr_stats}")
        print(f"[DEMO] Motion: {motion}")
    else:
        dbz_grid, ct_grid, grid_lat, grid_lon = fetch_real_radar_and_ir()
        curr_stats = analyze_cell_grid(dbz_grid, ct_grid, grid_lat, grid_lon)
        history = state.get("history", [])
        if history:
            last_entry = history[-1]
            prev_stats =
