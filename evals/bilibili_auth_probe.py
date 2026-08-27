from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

config = json.loads(Path("vendor/bilibili-home/.bilibili-mcp/config.json").read_text(encoding="utf-8"))
bvid = "BV1XPuo6uES8"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": f"https://www.bilibili.com/video/{bvid}",
    "Cookie": (
        f"SESSDATA={config['sessdata']}; "
        f"bili_jct={config['bili_jct']}; "
        f"DedeUserID={config['dedeuserid']}"
    ),
}
view_url = "https://api.bilibili.com/x/web-interface/view?" + urlencode({"bvid": bvid})
with urlopen(Request(view_url, headers=headers), timeout=20) as response:
    view = json.load(response)
cid = view["data"]["cid"]
player_url = "https://api.bilibili.com/x/player/v2?" + urlencode({"bvid": bvid, "cid": cid})
with urlopen(Request(player_url, headers=headers), timeout=20) as response:
    player = json.load(response)
tracks = player["data"]["subtitle"]["subtitles"]
print([(item.get("lan"), item.get("lan_doc")) for item in tracks])
subtitle_url = tracks[0]["subtitle_url"]
if subtitle_url.startswith("//"):
    subtitle_url = "https:" + subtitle_url
with urlopen(Request(subtitle_url, headers=headers), timeout=20) as response:
    raw = response.read()
body = json.loads(raw.decode("utf-8"))["body"]
print(" / ".join(item["content"] for item in body[:8]))
