"""Locked request ledger with conservative reservations and no automatic retries."""

from contextlib import contextmanager
import fcntl
from pathlib import Path

from pipeline import read, require, save


def charged(ledger):
    return sum(r.get("cost", r["reservation"]) for r in ledger["requests"])


@contextmanager
def locked(path):
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def call(path, payload, execute):
    path = Path(path)
    model = payload.get("model")
    require(model in ("google/gemini-3.8-flash", "google/gemini-3.1-flash-image"),
            "Budget ledger supports only the tested model pair.")
    require(0 < payload.get("max_tokens", 0) <= 8192, "Request token cap exceeds budget policy.")
    reservation = 0.60 if "image" in payload.get("modalities", []) else 0.05
    entry = {"model": model, "reservation": reservation, "status": "reserved"}
    with locked(path):
        ledger = read(path)
        require(charged(ledger) + reservation <= ledger["limit_usd"],
                "Budget headroom insufficient; no request sent.")
        index = len(ledger['requests'])
        ledger["requests"].append(entry)
        save(path, ledger)
    # If transport fails, retain the reservation: upstream cost may be unknown.
    result = execute()
    cost = result.get("usage", {}).get("cost")
    entry.update(status="completed", response_id=result.get("id"),
                 service_tier=result.get("service_tier"))
    if isinstance(cost, (int, float)) and cost >= 0:
        entry["cost"] = cost
    with locked(path):
        ledger = read(path)
        ledger['requests'][index] = entry
        save(path, ledger)
        require(charged(ledger) <= ledger["limit_usd"], "Reported spend exceeded budget; stop.")
    return result
