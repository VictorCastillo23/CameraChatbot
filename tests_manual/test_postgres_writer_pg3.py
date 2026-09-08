"""Manual verification for PR11 (psycopg2 -> psycopg v3 migration)'s highest-risk
sub-task: the `execute_values` replacement in `db.postgres_writer`.

Not a pytest test: this repo has no test runner, so this is a plain runnable
script using bare ``assert`` statements, printing a PASS/FAIL summary and
exiting 0/1. Run it directly:

    python tests_manual/test_postgres_writer_pg3.py

Why this file exists: `psycopg2.extras.execute_values` has NO direct v3
equivalent, and the wrong replacement (`cursor.executemany`) would silently
change behavior from ONE batched multi-row `INSERT ... VALUES (...),(...)...`
statement into N separate round-trip statements. `execute_values_compat()`
is the hand-written helper that preserves the single-statement, multi-row
shape under psycopg v3.

Design note (documented deviation from psycopg2's own internals): psycopg2's
`execute_values` renders each row via `cursor.mogrify(template, args)`,
INLINING literal values into the SQL text and executing with no separate
params. psycopg v3's default `Cursor` has NO `mogrify()` equivalent (dropped
in favor of server-side parameter binding; only the opt-in `ClientCursor`
has anything similar). `execute_values_compat()` therefore builds the
multi-row `VALUES (%s,...),(%s,...),...` SQL text by repeating the row
`template` per row, and passes the FLATTENED tuple of real parameter values
to `cur.execute(sql, flat_params)` -- one execute() call per page, real
parameterized binding (no manual literal-escaping), same externally
observable "one batched statement per page" property `execute_values` had.
Byte-identical SQL text to psycopg2's mogrify-inlined output is NOT a goal
(and isn't achievable without a from-scratch SQL literal quoter) -- the
property that matters and IS tested here is: exactly one `cur.execute()`
call per page, with the correct N-row `VALUES` clause and correctly
flattened params.

Uses only a hand-rolled fake cursor (`.execute()`/`.fetchall()` call
recorder) -- no real Postgres connection needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camerachatbot.db.postgres_writer import (  # noqa: E402
    execute_values_compat,
    execute_values_returning,
)

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
# fake psycopg-shaped cursor -- same pattern as test_zones.py/test_fase2_geometry.py
# ---------------------------------------------------------------------------

class _FakeCursor:
    """Records every `.execute()` call as `(sql, params)`. `.fetchall()`
    returns the next entry of `fetchall_results` (one per `.execute()` call,
    in order) -- lets a test simulate a distinct `RETURNING id` result set
    per page."""

    def __init__(self, fetchall_results=None):
        self._fetchall_results = list(fetchall_results or [])
        self.executed = []  # list[tuple[str, tuple]]

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        if not self._fetchall_results:
            return []
        return self._fetchall_results.pop(0)


# ---------------------------------------------------------------------------
# execute_values_compat(): single page -> ONE execute() call, N-row VALUES
# ---------------------------------------------------------------------------

def test_execute_values_compat_single_page_builds_one_statement():
    cur = _FakeCursor()
    rows = [(1, "a"), (2, "b")]
    execute_values_compat(
        cur, "INSERT INTO t (x, y) VALUES %s", rows, template="(%s, %s)", page_size=1000,
    )

    assert len(cur.executed) == 1, "must be exactly ONE cur.execute() call, not one per row"
    sql, params = cur.executed[0]
    assert sql == "INSERT INTO t (x, y) VALUES (%s, %s),(%s, %s)", sql
    assert params == (1, "a", 2, "b"), params


def test_execute_values_compat_empty_rows_is_a_no_op():
    cur = _FakeCursor()
    result = execute_values_compat(
        cur, "INSERT INTO t (x, y) VALUES %s", [], template="(%s, %s)", page_size=1000,
    )
    assert cur.executed == [], "empty rows must not issue any execute() call"
    assert result is None


# ---------------------------------------------------------------------------
# execute_values_compat(): pagination -> ONE execute() call PER PAGE, not
# per row (proves this is not a disguised `executemany`)
# ---------------------------------------------------------------------------

def test_execute_values_compat_paginates_into_one_execute_per_page():
    cur = _FakeCursor()
    rows = [(1,), (2,), (3,), (4,), (5,)]
    execute_values_compat(
        cur, "INSERT INTO t (x) VALUES %s", rows, template="(%s)", page_size=2,
    )

    assert len(cur.executed) == 3, "5 rows at page_size=2 -> 3 pages (2,2,1)"
    sql0, params0 = cur.executed[0]
    assert sql0 == "INSERT INTO t (x) VALUES (%s),(%s)", sql0
    assert params0 == (1, 2), params0

    sql1, params1 = cur.executed[1]
    assert sql1 == "INSERT INTO t (x) VALUES (%s),(%s)", sql1
    assert params1 == (3, 4), params1

    sql2, params2 = cur.executed[2]
    assert sql2 == "INSERT INTO t (x) VALUES (%s)", sql2
    assert params2 == (5,), params2


# ---------------------------------------------------------------------------
# execute_values_compat(returning=True): collects RETURNING ids per page
# ---------------------------------------------------------------------------

def test_execute_values_compat_returning_collects_ids_across_pages():
    cur = _FakeCursor(fetchall_results=[[(101,), (102,)], [(103,)]])
    rows = [(1,), (2,), (3,)]
    ids = execute_values_compat(
        cur, "INSERT INTO t (x) VALUES %s RETURNING id", rows, template="(%s)",
        page_size=2, returning=True,
    )

    assert len(cur.executed) == 2, "3 rows at page_size=2 -> 2 pages"
    assert ids == [101, 102, 103], ids


# ---------------------------------------------------------------------------
# execute_values_returning(): the real production wrapper -- appends
# "RETURNING id" and delegates to execute_values_compat(returning=True)
# ---------------------------------------------------------------------------

def test_execute_values_returning_appends_returning_id_and_collects_ids():
    cur = _FakeCursor(fetchall_results=[[(7,), (8,)]])
    rows = [(10, 20), (30, 40)]
    ids = execute_values_returning(cur, "INSERT INTO t (a, b) VALUES %s", rows, template="(%s, %s)")

    assert len(cur.executed) == 1
    sql, params = cur.executed[0]
    assert sql == "INSERT INTO t (a, b) VALUES %s RETURNING id".replace(
        "%s RETURNING", "(%s, %s),(%s, %s) RETURNING"
    ), sql
    assert params == (10, 20, 30, 40), params
    assert ids == [7, 8], ids


def test_execute_values_returning_empty_rows_returns_empty_list_without_executing():
    cur = _FakeCursor()
    ids = execute_values_returning(cur, "INSERT INTO t (a) VALUES %s", [], template="(%s)")
    assert ids == []
    assert cur.executed == []


def main():
    check(
        "execute_values_compat(): single page -> ONE execute() call, N-row VALUES SQL + flattened params",
        test_execute_values_compat_single_page_builds_one_statement,
    )
    check(
        "execute_values_compat(): empty rows -> no-op, zero execute() calls",
        test_execute_values_compat_empty_rows_is_a_no_op,
    )
    check(
        "execute_values_compat(): pagination -> one execute() per page, not per row",
        test_execute_values_compat_paginates_into_one_execute_per_page,
    )
    check(
        "execute_values_compat(returning=True): collects RETURNING ids across pages",
        test_execute_values_compat_returning_collects_ids_across_pages,
    )
    check(
        "execute_values_returning(): appends RETURNING id, delegates to execute_values_compat",
        test_execute_values_returning_appends_returning_id_and_collects_ids,
    )
    check(
        "execute_values_returning(): empty rows -> [] without issuing any execute() call",
        test_execute_values_returning_empty_rows_returns_empty_list_without_executing,
    )

    print("\n=== PR11 psycopg2 -> psycopg v3: execute_values replacement verification ===")
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if not ok:
            line += f" — {detail}"
        print(line)

    print(f"\n{n_pass}/{len(results)} checks passed")
    all_ok = n_pass == len(results)
    print("SUMMARY: PASS" if all_ok else "SUMMARY: FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
