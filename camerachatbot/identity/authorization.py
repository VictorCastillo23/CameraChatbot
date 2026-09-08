"""Fase 5 — authorization registry: answers "is this person ALLOWED", not
"have I seen this body before" (that second question belongs to the FAISS
gallery / `id_map.json`, via `identity.global_identity_service.
GlobalIdentityService`).

`AuthorizationRegistry` is an immutable snapshot of `authorized_identity`,
loaded ONCE per pipeline run -- batch semantics mean no invalidation is
needed, the same convention already established by
`geometry.homography.CameraCalibration`/`security.zones.load_zones()`.
Unlike those two, this registry is camera-independent (`authorized_identity`
has no `camera_id` column): `load()` takes no argument and should be called
regardless of whether the batch's camera has been calibrated yet -- Fase 5
is meant to be usable before any camera is calibrated.

**Fail-closed by design**: `is_authorized()` returns `False` for an unknown
or `None` person_global_id. That is deliberate, not a bug -- it is exactly
what makes "unenrolled person" detectable (see
`security.events.evaluate_unenrolled`). A fail-open default (treating an
unrecognized pid as authorized) would make this entire capability a no-op.

Deauthorization is `is_active=False` only (see `identity.enroll_person
--deactivate`): the person stays recognizable in the gallery/FAISS index and
now generates `unenrolled_person` events instead -- this is the intended
behavior, not a gap to "fix".
"""

from typing import Dict, Optional

from camerachatbot.db.postgres_writer import get_conn, SCHEMA


class AuthorizationRegistry:
    def __init__(self, rows: Dict[int, dict]):
        # Only ACTIVE rows are ever loaded into `rows` (see `load()`'s
        # `WHERE is_active` filter) -- so simple membership in this dict
        # already means "authorized", no extra `is_active` check needed at
        # lookup time.
        self._rows = dict(rows or {})

    @classmethod
    def load(cls) -> "AuthorizationRegistry":
        """Load every ACTIVE `authorized_identity` row into an immutable
        snapshot, keyed by `person_global_id`.

        Camera-independent: takes no `camera_id`, unlike `CameraCalibration.
        load(camera_id)`/`load_zones(camera_id)`. Callers should call this
        unconditionally, once per pipeline run, regardless of `camera_id`.
        """
        conn = get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET search_path TO {SCHEMA}")
                    cur.execute(
                        """
                        SELECT person_global_id, display_name, role, is_active, enrolled_at, notes
                        FROM authorized_identity
                        WHERE is_active
                        """
                    )
                    fetched = cur.fetchall()
            rows = {}
            for pid, display_name, role, is_active, enrolled_at, notes in (fetched or []):
                rows[int(pid)] = {
                    "person_global_id": int(pid),
                    "display_name": display_name,
                    "role": role,
                    "is_active": bool(is_active),
                    "enrolled_at": enrolled_at,
                    "notes": notes,
                }
            return cls(rows)
        finally:
            conn.close()

    def is_authorized(self, person_global_id: Optional[int]) -> bool:
        """Fail-closed: `None`, or any pid not present in this ACTIVE-only
        snapshot (never enrolled, or deauthorized), returns `False`.

        This is the crux of Fase 5's `unenrolled_person` detection -- an
        unrecognized or deauthorized person must read as "not authorized",
        never silently waved through.
        """
        if person_global_id is None:
            return False
        return int(person_global_id) in self._rows

    def get(self, person_global_id: Optional[int]) -> Optional[dict]:
        """The stored row dict for an authorized `person_global_id`, or
        `None` if unknown/deauthorized/`None`."""
        if person_global_id is None:
            return None
        return self._rows.get(int(person_global_id))
