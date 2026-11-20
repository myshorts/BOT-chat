import os, json, requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
LAT, LON = 20.20, 106.38          # xã Giao Phúc (ước lượng, bạn kiểm tra lại toạ độ)
PLACE = "Giao Phúc, Ninh Bình"
TOPIC = os.environ["NTFY_TOPIC"]
MODE = os.environ.get("MODE", "alert")
STATE_FILE = ".nowcast_state.json"
COOLDOWN_H = 2

PERIODS = [("Sáng", 6, 11), ("Trưa", 11, 14), ("Chiều", 14, 18), ("Tối", 18, 24)]
THUNDER = {95, 96, 99}

def fetch():
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON, "timezone": "Asia/Ho_Chi_Minh",
        "hourly": "temperature_2m,precipitation,precipitation_probability,"
                  "weather_code,cloud_cover,wind_gusts_10m,cape",
        "forecast_days": 2}, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    rows = []
    for i, t in enumerate(h["time"]):
        rows.append({k: h[k][i] for k in h if k != "time"} |
                    {"t": datetime.fromisoformat(t).replace(tzinfo=TZ)})
    return rows

def send(title, msg, priority=3, tags=("cloud",)):
    requests.post("https://ntfy.sh/", json={
        "topic": TOPIC, "title": title, "message": msg,
        "priority": priority, "tags": list(tags)}, timeout=30).raise_for_status()

def describe(rows):
    rain = sum(r["precipitation"] or 0 for r in rows)
    pop = max((r["precipitation_probability"] or 0) for r in rows)
    cloud = sum(r["cloud_cover"] or 0 for r in rows) / len(rows)
    temp = [r["temperature_2m"] for r in rows]
    thunder = any(r["weather_code"] in THUNDER for r in rows)
    if thunder:
        s = "⛈ Dông"
    elif rain >= 10:
        s = "🌧 Mưa to"
    elif rain >= 1 or pop >= 60:
        s = "🌦 Có mưa"
    elif cloud < 30:
        s = "☀️ Nắng"
    elif cloud < 70:
        s = "⛅ Ít mây, nắng nhẹ"
    else:
        s = "☁️ Nhiều mây"
    return f"{s} | {min(temp):.0f}-{max(temp):.0f}°C | mưa {rain:.1f}mm | xs {pop}%"

def daily():
    now = datetime.now(TZ)
    today = [r for r in fetch() if r["t"].date() == now.date()]
    lines = []
    for name, a, b in PERIODS:
        seg = [r for r in today if a <= r["t"].hour < b]
        if seg:
            lines.append(f"{name} ({a}-{b % 24}h): {describe(seg)}")
    total = sum(r["precipitation"] or 0 for r in today)
    lines.append(f"\nTổng lượng mưa dự báo cả ngày: {total:.1f} mm")
    send(f"Thời tiết {PLACE} {now:%d/%m}", "\n".join(lines))

def load_state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}

def alert():
    now = datetime.now(TZ)
    nxt = [r for r in fetch() if now - timedelta(hours=1) <= r["t"] <= now + timedelta(hours=3)]
    found = {}
    thunder = [r for r in nxt if r["weather_code"] in THUNDER or
               ((r["cape"] or 0) > 1500 and (r["precipitation"] or 0) >= 1)]
    if thunder:
        found["thunder"] = f"⛈ Có khả năng dông lúc {thunder[0]['t']:%H:%M}"
    heavy = [r for r in nxt if (r["precipitation"] or 0) >= 7.6]
    if heavy:
        found["heavy"] = (f"🌧 Mưa lớn lúc {heavy[0]['t']:%H:%M}, "
                          f"đỉnh {max(r['precipitation'] for r in heavy):.1f} mm/giờ")
    gust = [r for r in nxt if (r["wind_gusts_10m"] or 0) >= 60]
    if gust:
        found["gust"] = f"💨 Gió giật mạnh ~{max(r['wind_gusts_10m'] for r in gust):.0f} km/h"

    state = load_state()
    sent = False
    for k, text in found.items():
        last = state.get(k)
        if last and now - datetime.fromisoformat(last) < timedelta(hours=COOLDOWN_H):
            continue
        send(f"⚠️ Cảnh báo {PLACE}", text, priority=5, tags=("warning",))
        state[k] = now.isoformat()
        sent = True
    if sent:
        json.dump(state, open(STATE_FILE, "w"))

if __name__ == "__main__":
    daily() if MODE == "daily" else alert()
