#!/usr/bin/env python3

###############################################################################
# system_info_db.py
###############################################################################
import os
import psutil
import subprocess
import time
from datetime import datetime
import pytz
import sys
import json
import sqlite3

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

copenhagen_tz = pytz.timezone('Europe/Copenhagen')
DB_FILE = 'system_info.db'

def init_db():
    """
    Creates (if not existing) a SQLite database file with a 'points' table
    for system statistics.
    """
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cpu_usage REAL,
            cpu_usage_per_core TEXT,
            memory_usage REAL,
            disk_usage REAL,
            cpu_temp REAL,
            gpu_temp REAL,
            gpu_usage TEXT,
            nvme_temp REAL,
            uptime REAL
        )
        ''')
        conn.commit()

def get_system_info():
    """
    Query CPU, memory, disk usage, and temps. Return as a dict.
    """
    info = {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "cpu_usage_per_core": psutil.cpu_percent(interval=1, percpu=True),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "cpu_temp": get_cpu_temp(),
        "gpu_temp": get_gpu_temp(),
        "gpu_usage": get_gpu_usage(),
        "nvme_temp": get_nvme_temp(),
        "uptime": psutil.boot_time()
    }
    return info

def get_cpu_temp():
    try:
        output = subprocess.check_output(["sensors"], text=True)
        for line in output.splitlines():
            if "Tctl" in line:
                return float(line.split()[1].replace("°C", ""))
        return None
    except:
        return None

def get_gpu_temp():
    try:
        output = subprocess.check_output(["sensors"], text=True)
        for line in output.splitlines():
            if "edge" in line:
                return float(line.split()[1].replace("°C", ""))
        return None
    except:
        return None

def get_nvme_temp():
    try:
        output = subprocess.check_output(["sensors"], text=True)
        for line in output.splitlines():
            if "Composite" in line:
                return float(line.split()[1].replace("°C", ""))
        return None
    except:
        return None

def get_gpu_usage():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu", "--format=csv,noheader"],
            text=True
        )
        gpu_info = []
        for line in output.strip().splitlines():
            index, name, usage = line.split(", ")
            gpu_info.append({
                "index": int(index),
                "name": name,
                "usage": int(usage.replace(" %", ""))  # Remove '%' -> integer
            })
        return gpu_info
    except Exception as e:
        log("Error fetching GPU usage:", e)
        return None

def collect_minute_data():
    """
    Collect data for 1 minute at ~1-second intervals.
    Return a list of dicts, each with a timestamp and system stats.
    """
    data_points = []
    start_time = time.time()
    log("\n=== Starting Data Collection ===")

    try:
        while time.time() - start_time < 60:
            current_time = datetime.now(copenhagen_tz).isoformat()
            info = get_system_info()
            info["timestamp"] = current_time
            data_points.append(info)

            log(f"\rCollecting... Points: {len(data_points)} | "
                f"CPU: {info['cpu_usage']:.1f}% | "
                f"MEM: {info['memory_usage']:.1f}%", end="")
            time.sleep(1)

        log("\nCollection complete!")
    except Exception as e:
        log(f"\nError during collection: {e}")

    return data_points

def save_points_to_db(points):
    """
    Insert all collected data points into the SQLite 'points' table.
    """
    log("\n=== Starting Save Operation ===")
    if not points:
        log("No new points to save.")
        return 0

    inserted = 0
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        for p in points:
            c.execute('''
                INSERT INTO points (
                    timestamp, cpu_usage, cpu_usage_per_core, memory_usage,
                    disk_usage, cpu_temp, gpu_temp, gpu_usage,
                    nvme_temp, uptime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                p['timestamp'],
                p['cpu_usage'],
                json.dumps(p['cpu_usage_per_core']),  # store as JSON string
                p['memory_usage'],
                p['disk_usage'],
                p['cpu_temp'],
                p['gpu_temp'],
                json.dumps(p['gpu_usage']),           # also store as JSON
                p['nvme_temp'],
                p['uptime']
            ))
            inserted += 1
        conn.commit()

    log(f"Inserted {inserted} new points into DB.")
    return inserted

def get_all_data():
    """
    Query every row from the DB, returning a list of dicts.
    """
    all_data = []
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            SELECT timestamp, cpu_usage, cpu_usage_per_core, memory_usage,
                   disk_usage, cpu_temp, gpu_temp, gpu_usage, nvme_temp, uptime
            FROM points
            ORDER BY timestamp
        ''')
        rows = c.fetchall()
        for row in rows:
            p = {
                "timestamp": row[0],
                "cpu_usage": row[1],
                "cpu_usage_per_core": json.loads(row[2]) if row[2] else None,
                "memory_usage": row[3],
                "disk_usage": row[4],
                "cpu_temp": row[5],
                "gpu_temp": row[6],
                "gpu_usage": json.loads(row[7]) if row[7] else None,
                "nvme_temp": row[8],
                "uptime": row[9]
            }
            all_data.append(p)
    return all_data

def produce_json_structure():
    """
    Return a dict with:
      'current' = the last ~19 points
      'historical' = everything in the DB
    """
    all_points = get_all_data()
    if not all_points:
        return {"current": [], "historical": []}

    last_n = 19
    historical = all_points
    current = all_points[-last_n:]  # last 19 entries
    return {
        "current": current,
        "historical": historical
    }

if __name__ == "__main__":
    # Example usage: run this script to do one minute of collection & save to DB.
    log("\n=== Script Started ===")
    init_db()
    minute_data = collect_minute_data()
    count_inserted = save_points_to_db(minute_data)
    data_struct = produce_json_structure()
    log(f"Just inserted {count_inserted} points. Total in DB: {len(data_struct['historical'])}")
    log("\n=== Script Completed ===\n")