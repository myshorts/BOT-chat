#!/usr/bin/env python3
"""
Phù Cát Air Base (VVPC / Trung đoàn 925) — Automated Weather Nowcast Monitor
Tích hợp:
- Radar mưa RainViewer (Universal Blue color table -> dBZ & rain rate mm/h)
- Vệ tinh hồng ngoại JMA Himawari IR (nhiệt độ đỉnh mây Tb °C)
- Quan trắc bề mặt METAR VVPC & Open-Meteo (gió, tầm nhìn, trần mây, CAPE)
- Extrapolation chuyển động ổ mây (hướng, tốc độ km/h, ETA tới tâm sân bay)
- Cảnh báo đa cấp độ (Level 0 - 5) qua ntfy kèm chống spam (cooldown)
"""

import argparse
import datetime as dt
import io
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

# ==============================================================================
# CẤU HÌNH TỌA ĐỘ & KHU VỰC GIÁM SÁT (PHÙ CÁT - VVPC / TRUNG ĐOÀN 925)
# ==============================================================================
AIRPORT_NAME = "Sân bay Phù Cát (VVPC / Trung đoàn 925)"
CENTER_LAT = 13.9550
CENTER_LON = 109.0422

INNER_RADIUS_KM = 15.0   # Vùng trời sân bay & các khu bay lân cận (15 km)
OUTER_RADIUS_KM = 30.0   # Vành đai cảnh báo sớm ổ dông tiếp cận (30 km)

STATE_FILE = ".nowcast_state.json"
USER_AGENT = "PhuCat-Nowcast-Monitor/1.0 (VVPC-925)"

# Khoảng thời gian tối thiểu (phút) giữa 2 lần gửi cùng một mức cảnh báo
COOLDOWN_MINUTES = {
    5: 15,   # Cấp 5: Nguy hiểm khẩn cấp (dông mạnh / gió giật / mưa rất to tại sân)
    4: 20,   # Cấp 4: Cảnh báo cao (ổ dông mạnh trong 15km hoặc đang lao thẳng vào sân)
    3: 30,   # Cấp 3: Cảnh báo trung bình (mưa vừa / dông trong vành đai 15-30km)
    2: 45,   # Cấp 2: Theo dõi (mưa nhẹ / mây đối lưu phát triển)
    1: 60,   # Cấp 1: Bản tin định kỳ đầu giờ khi trời tốt
}


# ==============================================================================
# HÀM TOÁN HỌC & TỌA ĐỘ ĐỊA LÝ
# ==============================================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compass_dir_vi(deg: float) -> str:
    dirs = [
        "Bắc", "Bắc Đông Bắc", "Đông Bắc", "Đông Đông Bắc",
        "Đông", "Đông Đông Nam", "Đông Nam", "Nam Đông Nam",
        "Nam", "Nam Tây Nam", "Tây Nam", "Tây Tây Nam",
        "Tây", "Tây Tây Bắc", "Tây Bắc", "Bắc Tây Bắc",
    ]
    idx = int((deg + 11.25) // 22.5) % 16
    return dirs[idx]


def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int, float, float]:
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile_f = (lon + 180.0) / 360.0 * n
    ytile_f = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    xtile = int(xtile_f)
    ytile = int(ytile_f)
    px = (xtile_f - xtile) * 256.0
    py = (ytile_f - ytile) * 256.0
    return xtile, ytile, px, py


def tile_pixel_to_latlon(xtile: int, ytile: int, px: float, py: float, zoom: int) -> Tuple[float, float]:
    n = 2.0 ** zoom
    lon = (xtile + px / 256.0) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (ytile + py / 256.0) / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


# ==============================================================================
# QUẢN LÝ TRẠNG THÁI (STATE & COOLDOWN)
# ==============================================================================
def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"history": [], "last_sent": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("history", [])
                data.setdefault("last_sent", {})
                return data
    except Exception:
        pass
    return {"history": [], "last_sent": {}}


def save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[WARN] Không thể lưu state: {exc}", file=sys.stderr)


# ==============================================================================
# 1. RADAR RAINVIEWER (UNIVERSAL BLUE PALETTE -> dBZ)
# ==============================================================================
# Bảng màu chuẩn RainViewer Scheme 2 (Universal Blue) ánh xạ RGB -> dBZ
RAINVIEWER_PALETTE = [
    ((136, 221, 238), 10.0),
    ((0, 153, 255), 15.0),
    ((0, 102, 204), 20.0),
    ((0, 187, 0), 25.0),
    ((0, 136, 0), 30.0),
    ((255, 238, 0), 35.0),
    ((255, 170, 0), 40.0),
    ((255, 102, 0), 45.0),
    ((238, 0, 0), 50.0),
    ((187, 0, 0), 55.0),
    ((255, 0, 255), 60.0),
    ((153, 51, 204), 65.0),
]


def rgb_array_to_dbz(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Chuyển đổi mảng ảnh RGBA RainViewer sang ma trận phản hồi vô tuyến dBZ."""
    h, w, _ = rgb.shape
    dbz = np.zeros((h, w), dtype=np.float32)
    mask = alpha > 20
    if not np.any(mask):
        return dbz

    valid_rgb = rgb[mask].astype(np.float32)
    palette_colors = np.array([c[0] for c in RAINVIEWER_PALETTE], dtype=np.float32)
    palette_dbz = np.array([c[1] for c in RAINVIEWER_PALETTE], dtype=np.float32)

    # Tính khoảng cách màu Euclidean trong không gian RGB
    diff = valid_rgb[:, None, :] - palette_colors[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    nearest_idx = np.argmin(dist_sq, axis=1)
    min_dist_sq = dist_sq[np.arange(len(valid_rgb)), nearest_idx]

    matched_dbz = palette_dbz[nearest_idx]
    # Loại bỏ nhiễu màu không thuộc bảng radar
    matched_dbz[min_dist_sq > 8500.0] = 0.0
    dbz[mask] = matched_dbz
    return dbz


def dbz_to_rain_rate_mmh(dbz: float) -> float:
    """Công thức Marshall-Palmer Z = 200 * R^1.6."""
    if dbz < 10.0:
        return 0.0
    z = 10.0 ** (dbz / 10.0)
    return (z / 200.0) ** (1.0 / 1.6)


def fetch_rainviewer_radar() -> Dict[str, Any]:
    """Lấy khung radar mới nhất từ RainViewer API và quét vùng 15km & 30km quanh Phù Cát."""
    result: Dict[str, Any] = {
        "available": False,
        "radar_time_utc": None,
        "max_dbz_15km": 0.0,
        "mean_dbz_15km": 0.0,
        "coverage_15km_pct": 0.0,
        "max_dbz_30km": 0.0,
        "coverage_30km_pct": 0.0,
        "strongest_cell": None,
        "nearest_rain_cell": None,
    }
    try:
        resp = requests.get(
            "https://api.rainviewer.com/public/weather-maps.json",
            headers={"User-Agent": USER_AGENT},
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
        host = data.get("host", "https://tilecache.rainviewer.com")
        past = data.get("radar", {}).get("past", [])
        if not past:
            return result

        latest = past[-1]
        path = latest["path"]
        ts = latest["time"]
        result["radar_time_utc"] = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()

        # Zoom 8 (~611m/pixel tại vĩ độ 14°N) -> ghép lưới 3x3 tile quanh Phù Cát
        zoom = 8
        cx, cy, _, _ = latlon_to_tile(CENTER_LAT, CENTER_LON, zoom)
        canvas = Image.new("RGBA", (256 * 3, 256 * 3), (0, 0, 0, 0))

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tx, ty = cx + dx, cy + dy
                tile_url = f"{host}{path}/256/{zoom}/{tx}/{ty}/2/1_1.png"
                t_resp = requests.get(tile_url, headers={"User-Agent": USER_AGENT}, timeout=10)
                if t_resp.status_code == 200 and t_resp.content:
                    tile_img = Image.open(io.BytesIO(t_resp.content)).convert("RGBA")
                    canvas.paste(tile_img, ((dx + 1) * 256, (dy + 1) * 256))

        arr = np.array(canvas)
        dbz_grid = rgb_array_to_dbz(arr[:, :, :3], arr[:, :, 3])

        # Lấy mẫu theo bước 2 pixel để tính toán nhanh và chính xác
        inner_vals: List[float] = []
        outer_vals: List[float] = []
        inner_total = 0
        outer_total = 0

        strongest_cell: Optional[Dict[str, Any]] = None
        nearest_cell: Optional[Dict[str, Any]] = None

        top_tx, top_ty = cx - 1, cy - 1
        for py in range(0, 768, 2):
            for px in range(0, 768, 2):
                lat, lon = tile_pixel_to_latlon(top_tx, top_ty, float(px), float(py), zoom)
                dist = haversine_km(CENTER_LAT, CENTER_LON, lat, lon)
                if dist > OUTER_RADIUS_KM:
                    continue

                val = float(dbz_grid[py, px])
                outer_total += 1
                if val >= 15.0:
                    outer_vals.append(val)

                if dist <= INNER_RADIUS_KM:
                    inner_total += 1
                    if val >= 15.0:
                        inner_vals.append(val)

                if val >= 15.0:
                    brg = bearing_deg(CENTER_LAT, CENTER_LON, lat, lon)
                    cell_info = {
                        "lat": round(lat, 4),
                        "lon": round(lon, 4),
                        "dbz": round(val, 1),
                        "distance_km": round(dist, 1),
                        "bearing_deg": round(brg, 0),
                        "direction_vi": compass_dir_vi(brg),
                    }
                    if strongest_cell is None or val > strongest_cell["dbz"] or (
                        val == strongest_cell["dbz"] and dist < strongest_cell["distance_km"]
                    ):
                        strongest_cell = cell_info

                    if nearest_cell is None or dist < nearest_cell["distance_km"]:
                        nearest_cell = cell_info

        result["available"] = True
        if inner_total > 0:
            result["coverage_15km_pct"] = round(len(inner_vals) / inner_total * 100.0, 1)
        if outer_total > 0:
            result["coverage_30km_pct"] = round(len(outer_vals) / outer_total * 100.0, 1)

        if inner_vals:
            result["max_dbz_15km"] = round(max(inner_vals), 1)
            result["mean_dbz_15km"] = round(sum(inner_vals) / len(inner_vals), 1)
        if outer_vals:
            result["max_dbz_30km"] = round(max(outer_vals), 1)

        result["strongest_cell"] = strongest_cell
        result["nearest_rain_cell"] = nearest_cell
    except Exception as exc:
        print(f"[WARN] Lỗi đọc RainViewer radar: {exc}", file=sys.stderr)

    return result


# ==============================================================================
# 2. VỆ TINH HỒNG NGOẠI JMA HIMAWARI IR (NHIỆT ĐỘ ĐỈNH MÂY Tb °C)
# ==============================================================================
def fetch_jma_ir_cloud_top() -> Dict[str, Any]:
    """Lấy nhiệt độ đỉnh mây IR từ JMA Himawari (khu vực SE2 - Biển Đông / Việt Nam)."""
    result: Dict[str, Any] = {
        "available": False,
        "min_tb_c": None,
        "mean_tb_c": None,
        "convective_status": "Không xác định",
    }
    try:
        meta_url = "https://www.jma.go.jp/bosai/himawari/data/satimg/targetTimes_fd.json"
        resp = requests.get(meta_url, headers={"User-Agent": USER_AGENT}, timeout=12)
        resp.raise_for_status()
        times = resp.json()
        if not times:
            return result

        latest = times[-1]
        basetime = latest["basetime"]
        validtime = latest["validtime"]

        # Zoom 4 trên lưới Himawari JMA SE2 phủ khu vực Việt Nam
        zoom = 4
        xtile, ytile, px, py = latlon_to_tile(CENTER_LAT, CENTER_LON, zoom)
        tile_url = (
            f"https://www.jma.go.jp/bosai/himawari/data/satimg/{basetime}/fd/"
            f"{validtime}/B13/TBB/{zoom}/{xtile}/{ytile}.jpg"
        )
        img_resp = requests.get(tile_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if img_resp.status_code != 200 or not img_resp.content:
            return result

        img = Image.open(io.BytesIO(img_resp.content)).convert("L")
        arr = np.array(img, dtype=np.float32)

        cx_i, cy_i = int(round(px)), int(round(py))
        x0, x1 = max(0, cx_i - 4), min(256, cx_i + 5)
        y0, y1 = max(0, cy_i - 4), min(256, cy_i + 5)
        patch = arr[y0:y1, x0:x1]
        if patch.size == 0:
            return result

        # Ánh xạ thang xám ảnh IR JMA B13 sang nhiệt độ sáng đỉnh mây Tb (°C):
        # Pixel sáng (gần 255) tương ứng mây cao lạnh (-80°C), pixel tối (gần 0) tương ứng mặt đất ấm (+30°C)
        tb_patch = 30.0 - (patch / 255.0) * 110.0
        min_tb = float(np.min(tb_patch))
        mean_tb = float(np.mean(tb_patch))

        if min_tb <= -65.0:
            status = "Cb phát triển cực mạnh / tháp đối lưu xuyên tầng (<-65°C)"
        elif min_tb <= -52.0:
            status = "Mây vũ tích Cb đối lưu sâu hoạt động mạnh (-52°C đến -65°C)"
        elif min_tb <= -35.0:
            status = "Mây đối lưu trung bình / mây tầng cao dày (-35°C đến -52°C)"
        elif min_tb <= -15.0:
            status = "Mây tầng trung / mây dông đang hình thành (-15°C đến -35°C)"
        else:
            status = "Quang mây hoặc mây tầng thấp ấm (>-15°C)"

        result["available"] = True
        result["min_tb_c"] = round(min_tb, 1)
        result["mean_tb_c"] = round(mean_tb, 1)
        result["convective_status"] = status
    except Exception as exc:
        print(f"[WARN] Lỗi đọc vệ tinh JMA IR: {exc}", file=sys.stderr)

    return result


# ==============================================================================
# 3. QUAN TRẮC BỀ MẶT METAR VVPC & OPEN-METEO
# ==============================================================================
def fetch_station_weather_and_metar() -> Dict[str, Any]:
    """Lấy bản tin METAR mới nhất của VVPC và thông số bề mặt từ Open-Meteo."""
    weather: Dict[str, Any] = {
        "metar_raw": None,
        "temp_c": None,
        "dewpoint_c": None,
        "wind_dir_deg": None,
        "wind_speed_kt": None,
        "wind_gust_kt": None,
        "precip_mm": 0.0,
        "cape_jkg": None,
        "cloud_cover_pct": None,
        "visibility_km": None,
        "weather_code": None,
    }

    # 3A. Lấy METAR VVPC từ Aviation Weather Center (NOAA)
    try:
        metar_url = "https://aviationweather.gov/api/data/metar?ids=VVPC&format=json"
        m_resp = requests.get(metar_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if m_resp.status_code == 200:
            m_data = m_resp.json()
            if isinstance(m_data, list) and m_data:
                latest_m = m_data[0]
                weather["metar_raw"] = latest_m.get("rawOb")
                if latest_m.get("temp") is not None:
                    weather["temp_c"] = float(latest_m["temp"])
                if latest_m.get("dewp") is not None:
                    weather["dewpoint_c"] = float(latest_m["dewp"])
                if latest_m.get("wdir") is not None and str(latest_m["wdir"]).isdigit():
                    weather["wind_dir_deg"] = int(latest_m["wdir"])
                if latest_m.get("wspd") is not None:
                    weather["wind_speed_kt"] = float(latest_m["wspd"])
                if latest_m.get("wgst") is not None:
                    weather["wind_gust_kt"] = float(latest_m["wgst"])
    except Exception as exc:
        print(f"[WARN] Lỗi đọc METAR VVPC: {exc}", file=sys.stderr)

    # 3B. Bổ sung & đối chiếu từ Open-Meteo (tọa độ Phù Cát)
    try:
        om_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={CENTER_LAT}&longitude={CENTER_LON}"
            "&current=temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,rain,showers,"
            "weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,wind_gusts_10m"
            "&hourly=cape,visibility&wind_speed_unit=kn&forecast_days=1&timezone=Asia%2FHo_Chi_Minh"
        )
        o_resp = requests.get(om_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if o_resp.status_code == 200:
            o_data = o_resp.json()
            curr = o_data.get("current", {})
            if weather["temp_c"] is None and curr.get("temperature_2m") is not None:
                weather["temp_c"] = float(curr["temperature_2m"])
            if weather["dewpoint_c"] is None and curr.get("dew_point_2m") is not None:
                weather["dewpoint_c"] = float(curr["dew_point_2m"])
            if weather["wind_dir_deg"] is None and curr.get("wind_direction_10m") is not None:
                weather["wind_dir_deg"] = int(curr["wind_direction_10m"])
            if weather["wind_speed_kt"] is None and curr.get("wind_speed_10m") is not None:
                weather["wind_speed_kt"] = float(curr["wind_speed_10m"])
            if weather["wind_gust_kt"] is None and curr.get("wind_gusts_10m") is not None:
                weather["wind_gust_kt"] = float(curr["wind_gusts_10m"])

            weather["precip_mm"] = float(curr.get("precipitation") or 0.0)
            weather["cloud_cover_pct"] = curr.get("cloud_cover")
            weather["weather_code"] = curr.get("weather_code")

            hourly = o_data.get("hourly", {})
            capes = hourly.get("cape", [])
            vis_list = hourly.get("visibility", [])
            now_hour = dt.datetime.now().hour
            if capes and now_hour < len(capes) and capes[now_hour] is not None:
                weather["cape_jkg"] = float(capes[now_hour])
            if vis_list and now_hour < len(vis_list) and vis_list[now_hour] is not None:
                weather["visibility_km"] = round(float(vis_list[now_hour]) / 1000.0, 1)
    except Exception as exc:
        print(f"[WARN] Lỗi đọc Open-Meteo: {exc}", file=sys.stderr)

    return weather


# ==============================================================================
# 4. NGOẠI SUY CHUYỂN ĐỘNG Ổ MÂY (MOTION EXTRAPOLATION)
# ==============================================================================
def extrapolate_motion(
    prev_cell: Optional[Dict[str, Any]],
    curr_cell: Optional[Dict[str, Any]],
    dt_minutes: float,
) -> Optional[Dict[str, Any]]:
    """Tính vận tốc, hướng di chuyển ổ mây và ước tính thời gian tới sân bay (ETA)."""
    if not prev_cell or not curr_cell or dt_minutes <= 0:
        return None

    moved_km = haversine_km(
        prev_cell["lat"], prev_cell["lon"],
        curr_cell["lat"], curr_cell["lon"],
    )
    if moved_km < 0.8:
        return {
            "speed_kmh": 0.0,
            "move_dir_deg": None,
            "move_dir_vi": "Ít di chuyển / tại chỗ",
            "approaching": curr_cell["distance_km"] <= INNER_RADIUS_KM,
            "eta_min": 0 if curr_cell["distance_km"] <= 5.0 else None,
        }

    speed_kmh = moved_km / (dt_minutes / 60.0)
    if speed_kmh > 110.0:
        # Bỏ qua bước nhảy giả do đổi tâm ổ mây khác nhau
        return None

    move_brg = bearing_deg(
        prev_cell["lat"], prev_cell["lon"],
        curr_cell["lat"], curr_cell["lon"],
    )
    to_airport_brg = bearing_deg(
        curr_cell["lat"], curr_cell["lon"],
        CENTER_LAT, CENTER_LON,
    )
    angle_diff = abs((move_brg - to_airport_brg + 180.0) % 360.0 - 180.0)
    approaching = angle_diff <= 45.0 or (curr_cell["distance_km"] < prev_cell["distance_km"] - 1.0)

    eta_min: Optional[int] = None
    if approaching and speed_kmh >= 5.0:
        eta_min = max(1, int(round((curr_cell["distance_km"] / speed_kmh) * 60.0)))

    return {
        "speed_kmh": round(speed_kmh, 1),
        "move_dir_deg": round(move_brg, 0),
        "move_dir_vi": compass_dir_vi(move_brg),
        "approaching": approaching,
        "eta_min": eta_min,
    }


# ==============================================================================
# 5. ĐÁNH GIÁ MỨC ĐỘ CẢNH BÁO & SOẠN BẢN TIN NOWCAST
# ==============================================================================
def evaluate_alert(
    curr_stats: Dict[str, Any],
    prev_stats: Optional[Dict[str, Any]],
    motion: Optional[Dict[str, Any]],
    weather: Dict[str, Any],
    now_dt: dt.datetime,
) -> Optional[Dict[str, Any]]:
    radar = curr_stats.get("radar", {})
    ir = curr_stats.get("ir", {})

    max15 = float(radar.get("max_dbz_15km") or 0.0)
    cov15 = float(radar.get("coverage_15km_pct") or 0.0)
    max30 = float(radar.get("max_dbz_30km") or 0.0)
    strong_cell = radar.get("strongest_cell")
    near_cell = radar.get("nearest_rain_cell")

    min_tb = ir.get("min_tb_c")
    gust_kt = float(weather.get("wind_gust_kt") or 0.0)
    wspd_kt = float(weather.get("wind_speed_kt") or 0.0)
    precip_mm = float(weather.get("precip_mm") or 0.0)
    cape = float(weather.get("cape_jkg") or 0.0)
    wcode = int(weather.get("weather_code") or 0)

    has_thunder_code = wcode in (95, 96, 99)
    approaching = bool(motion and motion.get("approaching"))
    eta_min = motion.get("eta_min") if motion else None

    # Phân cấp mức độ nguy hiểm (Level 5 -> Level 1)
    level = 0
    key = "level_0"
    headline = ""

    if max15 >= 45.0 or (max15 >= 40.0 and gust_kt >= 25.0) or (has_thunder_code and max15 >= 35.0):
        level = 5
        key = "level_5_severe"
        headline = "CẢNH BÁO KHẨN CẤP: DÔNG MẠNH / MƯA RẤT TO TẠI VÙNG TRỜI PHÙ CÁT"
    elif (
        max15 >= 35.0
        or (max30 >= 45.0 and approaching)
        or (min_tb is not None and min_tb <= -58.0 and max30 >= 30.0)
        or gust_kt >= 28.0
    ):
        level = 4
        key = "level_4_high"
        headline = "CẢNH BÁO CAO: Ổ DÔNG MẠNH TRONG 15KM HOẶC ĐANG TIẾP CẬN VVPC"
    elif (
        max15 >= 25.0
        or max30 >= 35.0
        or (min_tb is not None and min_tb <= -45.0 and (max30 >= 20.0 or cape >= 1200.0))
        or gust_kt >= 22.0
    ):
        level = 3
        key = "level_3_moderate"
        headline = "CẢNH BÁO TRUNG BÌNH: MƯA RÀO / ĐỐI LƯU HOẠT ĐỘNG QUANH PHÙ CÁT"
    elif max15 >= 15.0 or max30 >= 20.0 or precip_mm >= 0.5:
        level = 2
        key = "level_2_watch"
        headline = "THEO DÕI THỜI TIẾT: XUẤT HIỆN MƯA NHẸ / MÂY ĐỐI LƯU LÂN CẬN"
    elif now_dt.minute < 10:
        # Gửi bản tin xác nhận trạng thái trời tốt vào đầu mỗi giờ
        level = 1
        key = f"level_1_hourly_{now_dt.strftime('%Y%m%d_%H')}"
        headline = "BẢN TIN ĐỊNH KỲ: THỜI TIẾT KHU VỰC PHÙ CÁT ỔN ĐỊNH"
    else:
        return None

    # Soạn nội dung chi tiết chuẩn nghiệp vụ khí tượng hàng không
    lines: List[str] = [
        f"Thời điểm: {now_dt.strftime('%H:%M %d/%m/%Y')} (Giờ địa phương)",
        f"Khu vực: {AIRPORT_NAME} (Bán kính giám sát 15km / 30km)",
        "",
        "1. RADAR THỜI TIẾT (RAINVIEWER):",
    ]

    if radar.get("available"):
        lines.append(
            f"- Trong 15km: Max {max15:.0f} dBZ (~{dbz_to_rain_rate_mmh(max15):.1f} mm/h), "
            f"độ phủ mưa {cov15:.1f}%"
        )
        lines.append(
            f"- Trong 30km: Max {max30:.0f} dBZ, độ phủ mưa {radar.get('coverage_30km_pct', 0.0):.1f}%"
        )
        if strong_cell:
            lines.append(
                f"- Tâm phản hồi mạnh nhất: {strong_cell['dbz']:.0f} dBZ cách sân bay "
                f"{strong_cell['distance_km']:.1f} km về hướng {strong_cell['direction_vi']} "
                f"({strong_cell['bearing_deg']:.0f}°)"
            )
        if near_cell and near_cell != strong_cell:
            lines.append(
                f"- Vùng mưa gần nhất: {near_cell['dbz']:.0f} dBZ cách {near_cell['distance_km']:.1f} km "
                f"hướng {near_cell['direction_vi']}"
            )
        if prev_stats and prev_stats.get("radar", {}).get("available"):
            prev_max15 = float(prev_stats["radar"].get("max_dbz_15km") or 0.0)
            delta = max15 - prev_max15
            if abs(delta) >= 3.0:
                trend_str = f"tăng +{delta:.0f} dBZ" if delta > 0 else f"giảm {delta:.0f} dBZ"
                lines.append(f"- Xu thế 10 phút qua (15km): {trend_str}")
    else:
        lines.append("- Dữ liệu radar tạm thời gián đoạn, đang theo dõi qua vệ tinh & mặt đất.")

    if motion:
        lines.append("")
        lines.append("2. DỊCH CHUYỂN Ổ MÂY (EXTRAPOLATION):")
        if motion["speed_kmh"] > 0:
            eta_txt = f" | Dự kiến tới sân bay sau ~{eta_min} phút" if eta_min else ""
            approach_txt = "ĐANG HƯỚNG VỀ SÂN BAY" if approaching else "Di chuyển lệch hướng sân bay"
            lines.append(
                f"- Hướng di chuyển: {motion['move_dir_vi']} ({motion['move_dir_deg']:.0f}°), "
                f"tốc độ ~{motion['speed_kmh']:.0f} km/h ({approach_txt}{eta_txt})"
            )
        else:
            lines.append(f"- Trạng thái dịch chuyển: {motion['move_dir_vi']}")

    lines.append("")
    lines.append("3. VỆ TINH IR & QUAN TRẮC BỀ MẶT:")
    if ir.get("available") and min_tb is not None:
        lines.append(f"- Đỉnh mây JMA IR: Min Tb = {min_tb:.1f}°C ({ir['convective_status']})")

    wind_parts = []
    if weather.get("wind_dir_deg") is not None:
        wind_parts.append(f"hướng {weather['wind_dir_deg']}°")
    if wspd_kt > 0:
        wind_parts.append(f"tốc độ {wspd_kt:.0f} kt")
    if gust_kt > 0:
        wind_parts.append(f"giật {gust_kt:.0f} kt")
    if wind_parts:
        lines.append("- Gió bề mặt: " + ", ".join(wind_parts))

    thermo_parts = []
    if weather.get("temp_c") is not None:
        thermo_parts.append(f"T={weather['temp_c']:.1f}°C")
    if weather.get("dewpoint_c") is not None:
        thermo_parts.append(f"Td={weather['dewpoint_c']:.1f}°C")
    if weather.get("visibility_km") is not None:
        thermo_parts.append(f"Tầm nhìn={weather['visibility_km']:.1f} km")
    if cape > 0:
        thermo_parts.append(f"CAPE={cape:.0f} J/kg")
    if thermo_parts:
        lines.append("- Chỉ số khí tượng: " + " | ".join(thermo_parts))

    if weather.get("metar_raw"):
        lines.append(f"- METAR: {weather['metar_raw']}")

    priority_map = {5: "urgent", 4: "high", 3: "default", 2: "low", 1: "min"}
    tags_map = {
        5: "rotating_light,thunder_cloud_and_rain",
        4: "warning,cloud_with_lightning_and_rain",
        3: "umbrella,cloud_with_rain",
        2: "eyes,partly_sunny",
        1: "white_check_mark,airplane",
    }

    return {
        "level": level,
        "key": key,
        "title": headline,
        "body": "\n".join(lines),
        "priority": priority_map.get(level, "default"),
        "tags": tags_map.get(level, "partly_sunny"),
    }


def can_send_alert(alert: Dict[str, Any], state: Dict[str, Any], now_dt: dt.datetime) -> bool:
    level = int(alert["level"])
    key = alert["key"]
    last_sent_map = state.get("last_sent", {})
    last_ts_str = last_sent_map.get(key)
    if not last_ts_str:
        return True
    try:
        last_dt = dt.datetime.fromisoformat(last_ts_str)
        elapsed_min = (now_dt - last_dt).total_seconds() / 60.0
        cooldown = COOLDOWN_MINUTES.get(level, 30)
        return elapsed_min >= cooldown
    except Exception:
        return True


def send_ntfy(alert: Dict[str, Any]) -> None:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        raise RuntimeError("Biến môi trường NTFY_TOPIC đang trống. Hãy kiểm tra GitHub Secrets.")

    url = topic if topic.startswith("http://") or topic.startswith("https://") else f"https://ntfy.sh/{topic}"
    headers = {
        "Title": alert["title"].encode("utf-8"),
        "Priority": alert["priority"],
        "Tags": alert["tags"],
        "Content-Type": "text/plain; charset=utf-8",
    }
    resp = requests.post(url, data=alert["body"].encode("utf-8"), headers=headers, timeout=15)
    resp.raise_for_status()
    print(f"[OK] Đã gửi thông báo ntfy thành công (HTTP {resp.status_code}).")


# ==============================================================================
# ĐIỂM VÀO CHƯƠNG TRÌNH (MAIN)
# ==============================================================================
def build_demo_data() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    curr_stats = {
        "radar": {
            "available": True,
            "max_dbz_15km": 46.0,
            "mean_dbz_15km": 32.5,
            "coverage_15km_pct": 28.0,
            "max_dbz_30km": 50.0,
            "coverage_30km_pct": 22.0,
            "strongest_cell": {
                "lat": 13.89,
                "lon": 108.96,
                "dbz": 46.0,
                "distance_km": 11.4,
                "bearing_deg": 230,
                "direction_vi": "Tây Nam",
            },
            "nearest_rain_cell": {
                "lat": 13.92,
                "lon": 109.00,
                "dbz": 35.0,
                "distance_km": 5.8,
                "bearing_deg": 230,
                "direction_vi": "Tây Nam",
            },
        },
        "ir": {
            "available": True,
            "min_tb_c": -61.5,
            "mean_tb_c": -44.0,
            "convective_status": "Mây vũ tích Cb đối lưu sâu hoạt động mạnh (-52°C đến -65°C)",
        },
    }
    prev_stats = {
        "radar": {
            "available": True,
            "max_dbz_15km": 38.0,
            "strongest_cell": {
                "lat": 13.84,
                "lon": 108.91,
                "dbz": 38.0,
                "distance_km": 19.2,
                "bearing_deg": 228,
                "direction_vi": "Tây Nam",
            },
        }
    }
    motion = {
        "speed_kmh": 34.0,
        "move_dir_deg": 48,
        "move_dir_vi": "Đông Bắc",
        "approaching": True,
        "eta_min": 20,
    }
    weather = {
        "metar_raw": "VVPC 190500Z 23015G26KT 4000 +TSRA FEW015CB BKN080 26/24 Q1009",
        "temp_c": 26.0,
        "dewpoint_c": 24.0,
        "wind_dir_deg": 230,
        "wind_speed_kt": 15.0,
        "wind_gust_kt": 26.0,
        "precip_mm": 6.4,
        "cape_jkg": 1850.0,
        "cloud_cover_pct": 92,
        "visibility_km": 4.0,
        "weather_code": 95,
    }
    return curr_stats, prev_stats, motion, weather


def main() -> None:
    parser = argparse.ArgumentParser(description="Phù Cát Weather Nowcast Monitor")
    parser.add_argument("--demo", action="store_true", help="Chạy thử kịch bản dông giả lập")
    parser.add_argument("--dry", action="store_true", help="Chạy dữ liệu thật nhưng không gửi ntfy")
    args = parser.parse_args()

    tz_vn = dt.timezone(dt.timedelta(hours=7))
    now_dt = dt.datetime.now(tz=tz_vn)
    state = load_state()

    prev_stats: Optional[Dict[str, Any]] = None
    motion: Optional[Dict[str, Any]] = None

    if args.demo:
        curr_stats, prev_stats, motion, weather = build_demo_data()
    else:
        radar_data = fetch_rainviewer_radar()
        ir_data = fetch_jma_ir_cloud_top()
        curr_stats = {"radar": radar_data, "ir": ir_data}

        history = state.get("history", [])
        if history:
            last_entry = history[-1]
            prev_stats = last_entry.get("stats")
            try:
                prev_time = dt.datetime.fromisoformat(last_entry["timestamp"])
                dt_min = max(1.0, (now_dt - prev_time).total_seconds() / 60.0)
            except Exception:
                dt_min = 10.0

            prev_radar = (prev_stats or {}).get("radar", {})
            prev_cell = prev_radar.get("strongest_cell") or prev_radar.get("nearest_rain_cell")
            curr_cell = radar_data.get("strongest_cell") or radar_data.get("nearest_rain_cell")
            motion = extrapolate_motion(prev_cell, curr_cell, dt_minutes=dt_min)

        weather = fetch_station_weather_and_metar()

    alert = evaluate_alert(curr_stats, prev_stats, motion, weather, now_dt)

    if not alert:
        print("[INFO] Thời tiết Phù Cát ổn định (không nằm trong khung gửi bản tin đầu giờ).")
    else:
        print(f"=== [LEVEL {alert['level']}] {alert['title']} ===")
        print(alert["body"])
        if not args.demo and not args.dry:
            if can_send_alert(alert, state, now_dt):
                send_ntfy(alert)
                state.setdefault("last_sent", {})[alert["key"]] = now_dt.isoformat()
            else:
                print("[SKIP] Đang trong thời gian cooldown chống spam.")

    if not args.demo and not args.dry:
        history = state.get("history", [])
        history.append({"timestamp": now_dt.isoformat(), "stats": curr_stats})
        state["history"] = history[-12:]
        save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
