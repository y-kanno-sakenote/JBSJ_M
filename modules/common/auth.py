"""Authentication helpers: simple username/password store using PBKDF2.

This is intentionally lightweight and local-only. Passwords are stored as
hex(salt) and hex(hash) in a JSON file. For production use, replace with
proper identity provider (OAuth/OpenID, external DB, etc.).
"""
from __future__ import annotations
import json
import os
import hashlib
import hmac
from pathlib import Path
from typing import Dict, Tuple, Optional

USERS_PATH = Path(__file__).resolve().parent / "users.json"

def _ensure_users_file() -> None:
    if not USERS_PATH.exists():
        try:
            USERS_PATH.write_text(json.dumps({}))
        except Exception:
            pass

def _pbkdf2_hash(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)

def load_users() -> Dict[str, Dict[str, str]]:
    _ensure_users_file()
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_users(users: Dict[str, Dict[str, str]]) -> None:
    try:
        USERS_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def add_user(username: str, password: str) -> bool:
    """Add or update a user with hashed password. Returns True on success."""
    if not username or not password:
        return False
    users = load_users()
    salt = os.urandom(16)
    h = _pbkdf2_hash(password, salt)
    # initialize with default plan 'free' and empty favorites list
    users[username] = {"salt": salt.hex(), "hash": h.hex(), "plan": "free", "favorites": []}
    save_users(users)
    return True

def verify_user(username: str, password: str) -> bool:
    users = load_users()
    rec = users.get(username)
    if not rec:
        return False
    try:
        salt = bytes.fromhex(rec.get("salt", ""))
        stored = bytes.fromhex(rec.get("hash", ""))
    except Exception:
        return False
    calc = _pbkdf2_hash(password, salt)
    try:
        return hmac.compare_digest(calc, stored)
    except Exception:
        # fallback to constant-time compare if needed
        return calc == stored

def users_file_path() -> Path:
    _ensure_users_file()
    return USERS_PATH


def get_plan(username: str) -> str:
    """Return the plan for the user. Defaults to 'free' if missing or user not found."""
    users = load_users()
    rec = users.get(username)
    if not rec:
        return "free"
    return rec.get("plan", "free")


def set_plan(username: str, plan: str) -> bool:
    """Set the plan for an existing user. Returns True on success."""
    if plan not in ("free", "paid"):
        return False
    users = load_users()
    if username not in users:
        return False
    users[username]["plan"] = plan
    save_users(users)
    return True


def is_paid(username: str) -> bool:
    """Helper: True when user's plan is 'paid'."""
    return get_plan(username) == "paid"
