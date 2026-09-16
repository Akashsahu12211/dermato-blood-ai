import sqlite3
import datetime

DB_PATH = "blood_group_app_v2.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            gender TEXT,
            predicted_blood_group TEXT,
            abo_confidence REAL,
            rh_confidence REAL,
            pattern_type TEXT,
            ridge_density REAL,
            sift_count INTEGER,
            raw_b64 TEXT,
            enhanced_b64 TEXT,
            skeleton_b64 TEXT,
            sift_b64 TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_scan_record(record_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO scan_records (
            patient_id, patient_name, age, gender, predicted_blood_group,
            abo_confidence, rh_confidence, pattern_type, ridge_density, sift_count,
            raw_b64, enhanced_b64, skeleton_b64, sift_b64, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(record_data["patient_id"]),
        str(record_data["patient_name"]),
        int(record_data["age"]),
        str(record_data["gender"]),
        str(record_data["predicted_blood_group"]),
        float(record_data["abo_confidence"]),
        float(record_data["rh_confidence"]),
        str(record_data["pattern_type"]),
        float(record_data["ridge_density"]),
        int(record_data.get("sift_count", 0)),
        str(record_data["raw_b64"]),
        str(record_data["enhanced_b64"]),
        str(record_data["skeleton_b64"]),
        str(record_data.get("sift_b64", "")),
        ts
    ))
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id

def get_scan_by_id(scan_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scan_records WHERE id = ?', (scan_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0], "patient_id": row[1], "patient_name": row[2],
            "age": row[3], "gender": row[4], "predicted_blood_group": row[5],
            "abo_confidence": float(row[6]), "rh_confidence": float(row[7]),
            "pattern_type": row[8], "ridge_density": float(row[9]),
            "sift_count": row[10] if len(row) > 10 else 44,
            "raw_b64": row[11] if len(row) > 11 else row[10],
            "enhanced_b64": row[12] if len(row) > 12 else row[11],
            "skeleton_b64": row[13] if len(row) > 13 else row[12],
            "sift_b64": row[14] if len(row) > 14 else row[12],
            "timestamp": row[15] if len(row) > 15 else row[13]
        }
    return None

def fetch_all_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, patient_id, patient_name, age, gender, predicted_blood_group, abo_confidence, rh_confidence, timestamp
        FROM scan_records ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for r in rows:
        logs.append({
            "id": r[0], "patient_id": r[1], "patient_name": r[2], "age": r[3],
            "gender": r[4], "predicted_blood_group": r[5],
            "abo_confidence": round(float(r[6]) * 100, 1),
            "rh_confidence": round(float(r[7]) * 100, 1),
            "timestamp": r[8]
        })
    return logs
