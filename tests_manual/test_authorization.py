"""Manual verification for the Fase 5 (PR9) authorization-registry contract.

Not a pytest test: this repo has no test runner, so this is a plain
runnable script using bare ``assert`` statements, printing a PASS/FAIL
summary and exiting 0/1. Run it directly:

    python tests_manual/test_authorization.py

What it verifies (Fase 5 / PR9 task "AuthorizationRegistry"):

1. `identity.authorization.AuthorizationRegistry.is_authorized()` --
   `True` for a pid present in the snapshot, `False` for an unknown pid,
   `False` for `None` (fail-closed by design -- this is what makes
   "unenrolled person" detectable, see `security.events.evaluate_
   unenrolled`).
2. `AuthorizationRegistry.get()` -- returns the stored row dict for a known
   pid, `None` for unknown/`None`.
3. `AuthorizationRegistry.load()` -- row -> snapshot shape (fake
   psycopg-shaped connection/cursor, no real Postgres needed -- same
   pattern as `test_zones.py`'s `load_zones()` coverage), including the
   "no active rows" -> empty-registry-that-authorizes-nothing degrade.

Uses only synthetic in-memory data throughout -- no live Postgres
connection, and never touches the repo's real `gallery.index`/
`id_map.json`/`proto_store.npy`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import camerachatbot.identity.authorization as authorization_mod  # noqa: E402
from camerachatbot.identity.authorization import AuthorizationRegistry  # noqa: E402

results = []  # list[tuple[str, bool, str]]


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
    except AssertionError as e:
        results.append((name, False, f"AssertionError: {e}"))
    except Exception as e:  # noqa: BLE001 - report any unexpected exception as a failure
        results.append((name, False, f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------
# is_authorized() / get(): direct construction, no DB
# ---------------------------------------------------------------------------

def _registry(pids=(1, 2)):
    rows = {pid: {"person_global_id": pid, "display_name": f"P{pid}", "role": None,
                  "is_active": True, "enrolled_at": None, "notes": None} for pid in pids}
    return AuthorizationRegistry(rows)


def test_is_authorized_true_for_known_pid():
    reg = _registry()
    assert reg.is_authorized(1) is True
    assert reg.is_authorized(2) is True


def test_is_authorized_false_for_unknown_pid():
    reg = _registry()
    assert reg.is_authorized(999) is False


def test_is_authorized_false_for_none_pid():
    # Fail-closed: this is the crux of "unenrolled person" detection.
    reg = _registry()
    assert reg.is_authorized(None) is False


def test_is_authorized_false_on_empty_registry():
    reg = AuthorizationRegistry({})
    assert reg.is_authorized(1) is False
    assert reg.is_authorized(None) is False


def test_get_returns_row_for_known_pid():
    reg = _registry()
    row = reg.get(1)
    assert row is not None
    assert row["person_global_id"] == 1
    assert row["display_name"] == "P1"


def test_get_returns_none_for_unknown_or_none_pid():
    reg = _registry()
    assert reg.get(999) is None
    assert reg.get(None) is None


# ---------------------------------------------------------------------------
# load(): fake psycopg-shaped connection/cursor
# (same pattern as test_zones.py's load_zones() tests)
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, fetchall_result=None):
        self._fetchall_result = fetchall_result if fetchall_result is not None else []
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._fetchall_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, fetchall_result=None):
        self._cursor = _FakeCursor(fetchall_result)
        self.closed = False

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        self.closed = True


def _with_fake_get_conn(fake_conn, fn):
    original_get_conn = authorization_mod.get_conn
    authorization_mod.get_conn = lambda: fake_conn
    try:
        return fn()
    finally:
        authorization_mod.get_conn = original_get_conn


def test_load_returns_empty_registry_when_no_active_rows():
    fake_conn = _FakeConn(fetchall_result=[])
    registry = _with_fake_get_conn(fake_conn, lambda: AuthorizationRegistry.load())
    assert registry.is_authorized(1) is False
    assert fake_conn.closed is True, "load() must close its connection even on the no-rows path"


def test_load_happy_path_builds_registry_from_rows():
    import datetime
    enrolled_at = datetime.datetime(2026, 1, 1, 0, 0, 0)
    fake_rows = [
        (7, "Ana", "staff", True, enrolled_at, None),
        (8, "Beto", None, True, enrolled_at, "note"),
    ]
    fake_conn = _FakeConn(fetchall_result=fake_rows)
    registry = _with_fake_get_conn(fake_conn, lambda: AuthorizationRegistry.load())

    assert registry.is_authorized(7) is True
    assert registry.is_authorized(8) is True
    assert registry.is_authorized(9) is False

    row7 = registry.get(7)
    assert row7["display_name"] == "Ana"
    assert row7["role"] == "staff"
    assert row7["is_active"] is True

    row8 = registry.get(8)
    assert row8["notes"] == "note"
    assert fake_conn.closed is True


def main():
    check("is_authorized(): True for a known pid", test_is_authorized_true_for_known_pid)
    check("is_authorized(): False for an unknown pid", test_is_authorized_false_for_unknown_pid)
    check("is_authorized(): False for pid=None (fail-closed)", test_is_authorized_false_for_none_pid)
    check("is_authorized(): False for every pid on an empty registry", test_is_authorized_false_on_empty_registry)
    check("get(): returns the stored row for a known pid", test_get_returns_row_for_known_pid)
    check("get(): None for unknown/None pid", test_get_returns_none_for_unknown_or_none_pid)
    check("load(): empty registry (authorizes nothing) on no active rows", test_load_returns_empty_registry_when_no_active_rows)
    check("load(): happy path builds registry from rows", test_load_happy_path_builds_registry_from_rows)

    print("\n=== Fase 5 (PR9) authorization-registry contract verification ===")
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if not ok:
            line += f" — {detail}"
        print(line)

    print(f"\n{n_pass}/{len(results)} checks passed.")
    if n_pass != len(results):
        print("SUMMARY: FAIL")
        return 1
    print("SUMMARY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
