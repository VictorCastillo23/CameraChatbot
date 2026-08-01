import json
import os
import time
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any, Dict

import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import errors
from dotenv import load_dotenv

from camerachatbot import paths

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DATABASE'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD')
}
SCHEMA = "view"
# Ajustables
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1000))      # tamaño de cada insert batch
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 2))       # hilos para metadata/neighborhood
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))       # reintentos ante deadlock
INITIAL_BACKOFF = 0.1                                # segundos

# --------------------- utilidades ---------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def retry_on_deadlock(fn):
    """Decorator simple para reintentar en DeadlockDetected / SerializationFailure."""
    def wrapper(*args, **kwargs):
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except (errors.DeadlockDetected, errors.SerializationFailure) as e:
                # Rollback connection if provided via cursor inside fn
                conn = kwargs.get("_conn") or (args[0] if args and hasattr(args[0], "rollback") else None)
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(backoff)
                backoff *= 2
    return wrapper

# --------------------- schema / tablas ---------------------

def ensure_schema_and_tables(cur):
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    cur.execute(f"SET search_path TO {SCHEMA}")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS project (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE,
            bucket TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS camera (
            id SERIAL PRIMARY KEY,
            project_id INTEGER REFERENCES project(id),
            name TEXT,
            description TEXT,
            installed_at TIMESTAMP,
            video_duration INTERVAL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS video (
            id SERIAL PRIMARY KEY,
            camera_id INTEGER REFERENCES camera(id),
            key TEXT,
            start_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS key_frame (
            id SERIAL PRIMARY KEY,
            video_id INTEGER REFERENCES video(id),
            timestamp TIMESTAMP,
            size POINT,
            inference FLOAT,
            preprocess FLOAT,
            postprocess FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS object_class (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS object (
            id SERIAL PRIMARY KEY,
            key_frame_id INTEGER REFERENCES key_frame(id),
            object_class_id INTEGER REFERENCES object_class(id),
            confidence FLOAT,
            shape BOX,
            user_id INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_object_user_id ON object(user_id)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            id SERIAL PRIMARY KEY,
            object_id INTEGER REFERENCES object(id),
            metadata_id INTEGER,
            name TEXT,
            data_type TEXT,
            value TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS neighborhood (
            id SERIAL PRIMARY KEY,
            key_frame_id INTEGER REFERENCES key_frame(id),
            related_object_id INTEGER REFERENCES object(id),
            intersection FLOAT,
            x_alignment FLOAT,
            y_alignment FLOAT,
            relation TEXT,
            markdown TEXT,
            parent_object_id INTEGER
        )
    """)

# --------------------- helpers DB ---------------------

def _fetchone(cur, q, params=None):
    cur.execute(q, params or ())
    return cur.fetchone()

def _fetchall(cur, q, params=None):
    cur.execute(q, params or ())
    return cur.fetchall()

def execute_values_returning(cur, base_sql: str, rows: List[Tuple], template: str, page_size=1000):
    """Inserta rows por execute_values y devuelve lista de ids"""
    if not rows:
        return []
    sql = base_sql + " RETURNING id"
    execute_values(cur, sql, rows, template=template, page_size=page_size)
    return [r[0] for r in cur.fetchall()]

# --------------------- inserciones principales (ordén garantizado) ---------------------

def get_or_create_project_camera_video(cur, data) -> Tuple[int, int, int]:
    project_id = _fetchone(cur, "SELECT id FROM project WHERE name=%s", (data.get("project_name", "Proyecto 1"),))
    if project_id:
        project_id = project_id[0]
    else:
        project_id = _fetchone(cur, "INSERT INTO project (name, bucket, created_at) VALUES (%s,%s,NOW()) RETURNING id",
                               (data.get("project_name", "Proyecto 1"), data.get("bucket", "bucket")))[0]

    camera_id = _fetchone(cur, "SELECT id FROM camera WHERE project_id=%s AND name=%s", (project_id, data.get("camera_name", "Cámara 2")))
    if camera_id:
        camera_id = camera_id[0]
    else:
        camera_id = _fetchone(cur, """
            INSERT INTO camera (project_id, name, description, installed_at, video_duration, created_at)
            VALUES (%s,%s,%s,%s,%s,NOW()) RETURNING id
        """, (project_id, data.get("camera_name", "Cámara 2"), data.get("camera_description", ""), data.get("installed_at"), None))[0]

    video_id = _fetchone(cur,
                         "INSERT INTO video (camera_id, key, start_at, created_at) VALUES (%s,%s,%s,NOW()) RETURNING id",
                         (camera_id, data["video"]["key"], data["video"]["start_at"]))[0]
    return project_id, camera_id, video_id

def prepare_keyframe_rows(video_id, key_frames):
    rows = []
    for kf in key_frames:
        ts = kf["timestamp"]
        sx = kf["size"]["x"]
        sy = kf["size"]["y"]
        inf = kf["inference"]
        pre = kf["preprocess"]
        pos = kf["postprocess"]
        rows.append((video_id, ts, sx, sy, inf, pre, pos))
    return rows

def insert_keyframes_in_batches(cur, video_id, key_frames):
    rows = prepare_keyframe_rows(video_id, key_frames)
    base = "INSERT INTO key_frame (video_id, timestamp, size, inference, preprocess, postprocess, created_at) VALUES %s"
    template = "(%s, %s, POINT(%s,%s), %s, %s, %s, NOW())"
    ids = []
    for chunk_rows in chunks(rows, BATCH_SIZE):
        ids_chunk = execute_values_returning(cur, base, chunk_rows, template, page_size=1000)
        ids.extend(ids_chunk)
    return ids

def collect_classes(key_frames):
    s = set()
    for kf in key_frames:
        for obj in kf.get("objects", []):
            s.add(obj["class_name"])
    return sorted(s)

def get_or_create_object_classes(cur, class_names: List[str]) -> Dict[str, int]:
    if not class_names:
        return {}
    # Upsert strategy: insertar los que no existen con ON CONFLICT DO NOTHING, luego consultar todos
    rows = [(n,) for n in class_names]
    base = "INSERT INTO object_class (name, created_at) VALUES %s ON CONFLICT (name) DO NOTHING"
    template = "(%s, NOW())"
    # hacer en batches por si son muchos
    for chunk_rows in chunks(rows, BATCH_SIZE):
        execute_values(cur, base, chunk_rows, template=template, page_size=1000)
    # ahora seleccionar
    existing = _fetchall(cur, "SELECT id,name FROM object_class WHERE name = ANY(%s)", (class_names,))
    by_name = {name: oid for oid, name in existing}
    return by_name

def prepare_objects_rows(keyframe_ids, key_frames, class_id_by_name):
    rows = []
    obj_index = []
    for kf_idx, kf in enumerate(key_frames):
        kf_id = keyframe_ids[kf_idx]
        for oi, obj in enumerate(kf.get("objects", [])):
            cn = obj["class_name"]
            cid = class_id_by_name[cn]
            conf = obj.get("confidence")
            sh = obj.get("shape", {"x1":0,"y1":0,"x2":0,"y2":0})
            x1, y1, x2, y2 = sh.get("x1",0), sh.get("y1",0), sh.get("x2",0), sh.get("y2",0)
            uid = obj.get("user_id")
            rows.append((kf_id, cid, conf, x1, y1, x2, y2, uid))
            obj_index.append((kf_idx, oi))
    return rows, obj_index

def insert_objects_in_batches(cur, keyframe_ids, key_frames, class_id_by_name):
    rows, obj_index = prepare_objects_rows(keyframe_ids, key_frames, class_id_by_name)
    base = "INSERT INTO object (key_frame_id, object_class_id, confidence, shape, user_id, created_at) VALUES %s"
    template = "(%s, %s, %s, BOX(POINT(%s,%s), POINT(%s,%s)), %s, NOW())"
    ids = []
    for chunk_rows in chunks(rows, BATCH_SIZE):
        ids_chunk = execute_values_returning(cur, base, chunk_rows, template, page_size=1000)
        ids.extend(ids_chunk)
    return ids, obj_index

# --------------------- funciones que se paralelizan por keyframe ranges ---------------------

def build_object_maps(object_ids, obj_index, key_frames):
    counts = [len(kf.get("objects", [])) for kf in key_frames]
    obj_ids_by_kf = [([None] * c) for c in counts]
    user_map_by_kf = [dict() for _ in counts]
    for oid, (kf_idx, oi) in zip(object_ids, obj_index):
        obj_ids_by_kf[kf_idx][oi] = oid
        o = key_frames[kf_idx]["objects"][oi]
        uid = o.get("user_id")
        if uid is not None:
            try:
                user_map_by_kf[kf_idx][int(uid)] = oid
            except Exception:
                pass
    return obj_ids_by_kf, user_map_by_kf

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def resolve_ref(ref, kf_idx, obj_ids_by_kf, user_map_by_kf):
    if ref is None:
        return None
    val = to_int(ref)
    if val is None:
        return None
    umap = user_map_by_kf[kf_idx]
    if val in umap:
        return umap[val]
    arr = obj_ids_by_kf[kf_idx]
    n = len(arr)
    if 0 <= val < n:
        return arr[val]
    if 1 <= val <= n:
        return arr[val - 1]
    return None

# Insert metadata recursivo (por objeto). Se ejecuta dentro de la conexión del worker.
def insert_metadata_recursive(cur, obj_id, meta, parent_id=None):
    cur.execute("""
        INSERT INTO metadata (object_id, metadata_id, name, data_type, value, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        RETURNING id
    """, (obj_id, parent_id, meta.get("name"), meta.get("data_type"), meta.get("value")))
    mid = cur.fetchone()[0]
    for child in meta.get("children", []):
        insert_metadata_recursive(cur, obj_id, child, mid)

# Worker que inserta metadata y neighborhood para un rango de keyframes (indices)
@retry_on_deadlock
def worker_insert_meta_and_nb(start_idx: int, end_idx: int,
                              key_frames, keyframe_ids, obj_ids_by_kf, user_map_by_kf,
                              video_neighborhood):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {SCHEMA}")
                # metadata por objetos
                for kf_idx in range(start_idx, end_idx):
                    kf = key_frames[kf_idx]
                    for oi, obj in enumerate(kf.get("objects", [])):
                        oid = obj_ids_by_kf[kf_idx][oi]
                        metas = obj.get("metadata") or []
                        for meta in metas:
                            insert_metadata_recursive(cur, oid, meta)

                # neighborhood por keyframe (resuelto dentro del mismo rango)
                for kf_idx in range(start_idx, end_idx):
                    kf_id = keyframe_ids[kf_idx]
                    nb = key_frames[kf_idx].get("neighborhood") or {}
                    rels = nb.get("relations", []) if isinstance(nb, dict) else []
                    rows = []
                    for r in rels:
                        rid = resolve_ref(r.get("related_object_id"), kf_idx, obj_ids_by_kf, user_map_by_kf)
                        pid = resolve_ref(r.get("parent_object_id"), kf_idx, obj_ids_by_kf, user_map_by_kf)
                        rows.append((kf_id, rid, r.get("intersection"), r.get("x_alignment"),
                                     r.get("y_alignment"), r.get("relation", ""), r.get("markdown", ""), pid))
                    if rows:
                        base = """INSERT INTO neighborhood
                            (key_frame_id, related_object_id, intersection, x_alignment, y_alignment, relation, markdown, parent_object_id)
                            VALUES %s"""
                        template = "(%s,%s,%s,%s,%s,%s,%s,%s)"
                        execute_values(cur, base, rows, template=template, page_size=1000)

                # video-level neighborhood — lo manejamos solo en el worker 0 (para evitar duplicados)
                # NOTE: video_neighborhood puede repetirse; para simplicidad solo lo inserta el worker que tenga start_idx == 0
                if start_idx == 0 and video_neighborhood:
                    rows = []
                    for r in video_neighborhood:
                        pos = r.get("key_frame_id")
                        pos_i = to_int(pos)
                        if not pos_i:
                            continue
                        if not (1 <= pos_i <= len(keyframe_ids)):
                            continue
                        kf_idx = pos_i - 1
                        kf_id = keyframe_ids[kf_idx]
                        rid = resolve_ref(r.get("related_object_id"), kf_idx, obj_ids_by_kf, user_map_by_kf)
                        pid = resolve_ref(r.get("parent_object_id"), kf_idx, obj_ids_by_kf, user_map_by_kf)
                        rows.append((kf_id, rid, r.get("intersection"), r.get("x_alignment"),
                                     r.get("y_alignment"), r.get("relation", ""), r.get("markdown", ""), pid))
                    if rows:
                        base = """INSERT INTO neighborhood
                            (key_frame_id, related_object_id, intersection, x_alignment, y_alignment, relation, markdown, parent_object_id)
                            VALUES %s"""
                        template = "(%s,%s,%s,%s,%s,%s,%s,%s)"
                        execute_values(cur, base, rows, template=template, page_size=1000)
        conn.close()
    finally:
        try:
            conn.close()
        except Exception:
            pass

# --------------------- flujo principal ---------------------

def json_to_postgre(json_file: str) -> str:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) crear esquema y tablas en una transacción corta
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                ensure_schema_and_tables(cur)
        conn.close()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # 2) abrir transacción para project/camera/video + key_frames + object_classes + objects (fase síncrona)
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"SET search_path TO {SCHEMA}")
                # Proyecto / cámara / video
                project_id, camera_id, video_id = get_or_create_project_camera_video(cur, data)

                # key_frames -> insert en batches
                key_frames = data["video"].get("key_frames", [])
                keyframe_ids = insert_keyframes_in_batches(cur, video_id, key_frames)

                # object classes (upsert)
                class_names = collect_classes(key_frames)
                class_map = get_or_create_object_classes(cur, class_names)

                # objects -> en batches (síncrono): necesitamos los object ids en orden para los pasos siguientes
                object_ids, obj_index = insert_objects_in_batches(cur, keyframe_ids, key_frames, class_map)

                # Nota: no insertamos metadata/neighborhood aquí; lo hacemos en paralelo más abajo
            # commit implícito por with conn:
        conn.close()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # 3) construir mapas para resolución de referencias
    obj_ids_by_kf, user_map_by_kf = build_object_maps(object_ids, obj_index, key_frames)

    # 4) Paralelizar metadata y neighborhood por rangos de keyframes (cada worker su propia conexión)
    total_kf = len(key_frames)
    if total_kf == 0:
        return "OK (no keyframes)"
    # dividir en MAX_WORKERS rangos (evitar que dos hilos trabajen el mismo keyframe)
    workers = min(MAX_WORKERS, total_kf)
    kf_per_worker = ceil(total_kf / workers)
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        for i in range(workers):
            start_idx = i * kf_per_worker
            end_idx = min((i + 1) * kf_per_worker, total_kf)
            if start_idx >= end_idx:
                continue
            futures.append(exe.submit(worker_insert_meta_and_nb,
                                       start_idx, end_idx,
                                       key_frames, keyframe_ids, obj_ids_by_kf, user_map_by_kf,
                                       data["video"].get("neighborhood", [])))
        # esperar resultados y propagar excepciones si occuren
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                # si un worker falla irrecoveriblemente, lo subimos
                raise exc

    return "OK"

# --------------------- main ---------------------

if __name__ == "__main__":
    result = json_to_postgre(str(paths.VIDEO_SCHEMA_OUTPUT_PATH))
    print(result)
