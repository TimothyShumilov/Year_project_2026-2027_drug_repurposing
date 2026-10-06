"""Small, auditable public-data client with immutable request caches."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def fetch_json(url, payload=None, *, offline=False, timeout=90, attempts=3):
    body = None if payload is None else json.dumps(payload, sort_keys=True).encode()
    key = hashlib.sha256(url.encode() + (body or b"")).hexdigest()
    path = ROOT / "data" / "raw" / "cache" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())["response"]
    if offline:
        raise FileNotFoundError(f"Missing cached response for {url}")
    headers = {"User-Agent": "Mozilla/5.0 CrohnFeasibilityPilot/0.1", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.load(response)
            if isinstance(data, dict) and data.get("errors"):
                raise ValueError("GraphQL error: " + json.dumps(data["errors"])[:3000])
            record = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "url": url,
                      "request": payload, "response_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                      "response": data}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False))
            temporary.replace(path)
            return data
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"Public API HTTP {error.code}: {error.read().decode()[:3000]}") from error
            if attempt + 1 == attempts:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def graphql(query, variables=None, *, offline=False):
    return fetch_json(OT_URL, {"query": query, "variables": variables or {}}, offline=offline)["data"]


def save_json(name, data):
    path = ROOT / "data" / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
