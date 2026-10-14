"""
threat_core.py - shared building blocks for the Sentinel threat-detection project.

Both rullall.py (training pipelines) and api_server.py (REST API + dashboard)
import from here, so the model always sees features built the same way.

Contents
  - paths and theme colors
  - traffic simulator (synthetic network connections + 4 attack types)
  - feature engineering and preprocessing
  - severity levels, response playbook, plain-language reasons
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
IMAGES = OUTPUTS / "images"
ALL_DIRS = [DATA_RAW, DATA_PROCESSED, MODELS, OUTPUTS, IMAGES, ROOT / "docs"]

# ----------------------------------------------------------------- theme
# "Night watch" palette: deep ink-blue base, with colors that carry meaning
# (teal = safe, blue/amber/orange/coral = rising severity). The dashboard CSS
# and every matplotlib chart use these same values.
THEME = dict(
    bg="#0b1a2c", panel="#112640", raise_="#173250", line="#27476b",
    text="#e4eef8", muted="#8da6c1",
    safe="#3dd3b0", low="#5fa8ff", medium="#f4b942", high="#ff8a4c", critical="#ff4d5e",
)

# ------------------------------------------------------------- vocabulary
CLASSES = ["normal", "dos", "brute_force", "port_scan", "data_exfiltration"]
LABELS = {
    "normal": "Normal", "dos": "DoS flood", "brute_force": "Brute force",
    "port_scan": "Port scan", "data_exfiltration": "Data exfiltration",
    "unknown_anomaly": "Unusual behaviour",
}
SEVERITY = {
    "normal": "none", "port_scan": "medium", "brute_force": "high",
    "dos": "critical", "data_exfiltration": "critical", "unknown_anomaly": "low",
}
PLAYBOOK = {
    "normal": "No action needed. Traffic matches the learned baseline.",
    "dos": "Rate-limit or null-route the source, enable SYN cookies, and alert the network team.",
    "brute_force": "Lock the targeted account, block the source IP, and require MFA for remote login.",
    "port_scan": "Watch the source, tighten firewall rules on exposed ports, and look for follow-up exploits.",
    "data_exfiltration": "Isolate the host, review outbound destinations, and start incident response.",
    "unknown_anomaly": "Review manually. The behaviour is unusual but matches no known attack pattern.",
}

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "https", "ssh", "dns", "ftp", "smtp"]
NUMERIC_RAW = ["duration", "src_bytes", "dst_bytes", "pkts_per_sec",
               "conn_count", "failed_logins", "syn_error_rate", "unique_ports"]
CATEGORICAL = ["protocol", "service"]
RAW_FIELDS = NUMERIC_RAW + CATEGORICAL

NUMERIC_MODEL = ["duration", "pkts_per_sec", "conn_count", "failed_logins",
                 "syn_error_rate", "unique_ports", "log_src_bytes", "log_dst_bytes",
                 "log_bytes_per_sec", "log_bytes_ratio", "ports_per_conn"]
FEATURES = NUMERIC_MODEL + CATEGORICAL

# ------------------------------------------------------------- simulator
CLASS_MIX = {"normal": 0.70, "dos": 0.10, "brute_force": 0.07,
             "port_scan": 0.07, "data_exfiltration": 0.06}
STEALTH_RATE = 0.04  # share of attacks that look almost normal (keeps metrics realistic)

# Documentation-only IP ranges (RFC 5737) so nothing here maps to a real host.
ATTACKER_POOL = {
    "dos": ["203.0.113.14", "203.0.113.77", "198.51.100.23"],
    "brute_force": ["198.51.100.66", "203.0.113.201"],
    "port_scan": ["192.0.2.45", "192.0.2.46"],
    "data_exfiltration": ["10.0.2.88"],  # a compromised internal host
}
SERVERS = ["10.0.10.10", "10.0.10.11", "10.0.10.12", "10.0.10.20", "10.0.10.30"]


def _pick(rng, options, probs, m):
    return rng.choice(options, size=m, p=probs)


def _gen(cls: str, m: int, rng) -> dict:
    """Generate m connections whose statistics look like class `cls`."""
    ln = rng.lognormal
    if cls == "normal":
        return dict(duration=ln(0.5, 0.8, m), src_bytes=ln(6.5, 1.0, m), dst_bytes=ln(7.5, 1.2, m),
                    pkts_per_sec=ln(2.0, 0.6, m), conn_count=rng.poisson(5, m),
                    failed_logins=rng.poisson(0.1, m), syn_error_rate=rng.beta(1, 30, m),
                    unique_ports=rng.poisson(2, m) + 1,
                    protocol=_pick(rng, PROTOCOLS, [.70, .25, .05], m),
                    service=_pick(rng, SERVICES, [.30, .40, .02, .20, .03, .05], m))
    if cls == "dos":  # flood: many short, tiny, half-open connections
        return dict(duration=ln(-1.0, 0.5, m), src_bytes=ln(5.5, 0.6, m), dst_bytes=ln(3.0, 1.0, m),
                    pkts_per_sec=ln(6.5, 0.5, m), conn_count=rng.poisson(220, m),
                    failed_logins=np.zeros(m, dtype=int), syn_error_rate=rng.beta(20, 3, m),
                    unique_ports=rng.integers(1, 3, m),
                    protocol=_pick(rng, PROTOCOLS, [.60, .25, .15], m),
                    service=_pick(rng, SERVICES, [.50, .40, 0, .10, 0, 0], m))
    if cls == "brute_force":  # password guessing against ssh/ftp/smtp
        return dict(duration=ln(1.5, 0.5, m), src_bytes=ln(5.0, 0.5, m), dst_bytes=ln(5.0, 0.5, m),
                    pkts_per_sec=ln(3.2, 0.4, m), conn_count=rng.poisson(40, m),
                    failed_logins=rng.poisson(12, m) + 3, syn_error_rate=rng.beta(2, 12, m),
                    unique_ports=np.ones(m, dtype=int),
                    protocol=_pick(rng, PROTOCOLS, [1, 0, 0], m),
                    service=_pick(rng, SERVICES, [0, 0, .60, 0, .30, .10], m))
    if cls == "port_scan":  # probing many ports with little data
        return dict(duration=ln(-0.5, 0.4, m), src_bytes=ln(4.0, 0.4, m), dst_bytes=ln(3.5, 0.5, m),
                    pkts_per_sec=ln(4.0, 0.5, m), conn_count=rng.poisson(60, m),
                    failed_logins=np.zeros(m, dtype=int), syn_error_rate=rng.beta(8, 6, m),
                    unique_ports=rng.poisson(45, m) + 10,
                    protocol=_pick(rng, PROTOCOLS, [.85, .15, 0], m),
                    service=_pick(rng, SERVICES, np.full(6, 1 / 6), m))
    if cls == "data_exfiltration":  # a few long connections pushing out lots of data
        return dict(duration=ln(3.0, 0.6, m), src_bytes=ln(13.0, 0.8, m), dst_bytes=ln(5.0, 0.6, m),
                    pkts_per_sec=ln(3.0, 0.5, m), conn_count=rng.poisson(4, m),
                    failed_logins=np.zeros(m, dtype=int), syn_error_rate=rng.beta(1, 30, m),
                    unique_ports=rng.integers(1, 3, m),
                    protocol=_pick(rng, PROTOCOLS, [1, 0, 0], m),
                    service=_pick(rng, SERVICES, [0, .60, 0, .20, .20, 0], m))
    raise ValueError(f"unknown class {cls}")


def simulate_traffic(n: int = 20000, seed: int = 42, stealth: float = STEALTH_RATE) -> pd.DataFrame:
    """Build a labelled, time-stamped table of n synthetic network connections."""
    rng = np.random.default_rng(seed)
    counts = {c: int(n * p) for c, p in CLASS_MIX.items()}
    counts["normal"] += n - sum(counts.values())
    frames = []
    for cls, m in counts.items():
        data = _gen(cls, m, rng)
        if cls != "normal" and stealth > 0:
            hide = rng.random(m) < stealth
            if hide.any():
                soft = _gen("normal", int(hide.sum()), rng)
                for k in soft:
                    data[k][hide] = soft[k]
        df = pd.DataFrame(data)
        df["label"] = cls
        if cls == "normal":
            df["src_ip"] = [f"10.0.{a}.{b}" for a, b in zip(rng.integers(0, 4, m), rng.integers(2, 250, m))]
        else:
            df["src_ip"] = rng.choice(ATTACKER_POOL[cls], m)
        df["dst_ip"] = rng.choice(SERVERS, m)
        frames.append(df)
    df = pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    gaps = rng.exponential(2.0, len(df))
    start = pd.Timestamp.now().floor("s") - pd.to_timedelta(gaps.sum(), unit="s")
    df.insert(0, "timestamp", (start + pd.to_timedelta(np.cumsum(gaps), unit="s")).round("s"))
    return df


# ------------------------------------------------------ feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-scaled and ratio features that make attacks stand out."""
    out = df.copy()
    out["log_src_bytes"] = np.log1p(out["src_bytes"])
    out["log_dst_bytes"] = np.log1p(out["dst_bytes"])
    out["log_bytes_per_sec"] = np.log1p((out["src_bytes"] + out["dst_bytes"]) / (out["duration"] + 0.01))
    out["log_bytes_ratio"] = out["log_src_bytes"] - out["log_dst_bytes"]
    out["ports_per_conn"] = out["unique_ports"] / (out["conn_count"] + 1)
    return out


def make_preprocessor() -> ColumnTransformer:
    """Scale numeric columns, one-hot encode protocol and service."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_MODEL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


# ---------------------------------------------------- API-side helpers
def clean_record(data: dict) -> dict:
    """Validate one connection record sent to the API. Raises ValueError with a clear message."""
    rec = {}
    for f in NUMERIC_RAW:
        if f not in data:
            raise ValueError(f"Missing field: {f}")
        try:
            v = float(data[f])
        except (TypeError, ValueError):
            raise ValueError(f"{f} must be a number")
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{f} must be zero or greater")
        rec[f] = v
    if rec["syn_error_rate"] > 1:
        raise ValueError("syn_error_rate must be between 0 and 1")
    for f, options in (("protocol", PROTOCOLS), ("service", SERVICES)):
        v = str(data.get(f, "")).lower()
        if v not in options:
            raise ValueError(f"{f} must be one of: {', '.join(options)}")
        rec[f] = v
    return rec


def explain(rec: dict) -> list[str]:
    """Plain-language reasons an analyst can read next to a verdict."""
    why = []
    if rec["failed_logins"] >= 5:
        why.append(f"{int(rec['failed_logins'])} failed logins in one session")
    if rec["syn_error_rate"] > 0.5:
        why.append("Most connections were left half-open (SYN errors)")
    if rec["unique_ports"] >= 20:
        why.append(f"Touched {int(rec['unique_ports'])} distinct ports")
    if rec["pkts_per_sec"] > 300:
        why.append("Very high packet rate")
    if rec["conn_count"] > 100:
        why.append(f"{int(rec['conn_count'])} connections from the same source")
    if rec["src_bytes"] > 200_000:
        why.append("Large outbound transfer")
    return why
