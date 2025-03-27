#!/usr/bin/env python3

###############################################################################
# system_info_db.py
###############################################################################
import os
import psutil
import subprocess
import time
from datetime import datetime, timedelta
import pytz
import sys
import json
import sqlite3

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

copenhagen_tz = pytz.timezone('Europe/Copenhagen')
DB_FILE = 'system_info.db'

# Check if current time is within active hours (8am to 10pm Copenhagen time)
def is_active_hours():
    current_time = datetime.now(copenhagen_tz)
    hour = current_time.hour
    return 8 <= hour < 22  # Between 8am and 10pm

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

def produce_json_structure(historical_limit=10000, historical_offset=0):
    """
    Produce the JSON data structure with current and historical data
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get current data (last minute)
            try:
                cutoff_timestamp = (datetime.now(copenhagen_tz) - 
                                  timedelta(minutes=1)).isoformat()
                
                cursor.execute("SELECT * FROM points WHERE timestamp >= ?", (cutoff_timestamp,))
                current_points = [dict(row) for row in cursor.fetchall()]
                print(f"Retrieved {len(current_points)} current points")
            except Exception as e:
                print(f"Error fetching current data: {e}")
                current_points = []
            
            # Implement downsampling for very large queries
            downsample_factor = 1
            if historical_limit > 5000:
                # Calculate the downsample factor based on the requested limit
                total_points = cursor.execute("SELECT COUNT(*) FROM points").fetchone()[0]
                if total_points > 5000:
                    downsample_factor = max(1, total_points // 5000)
                    print(f"Downsampling with factor {downsample_factor} based on total {total_points} points")
            
            # Get historical data efficiently
            try:
                if downsample_factor > 1:
                    # Use the modulo of rowid for efficient downsampling
                    cursor.execute("""
                        SELECT * FROM points 
                        WHERE timestamp < ? AND rowid % ? = 0
                        ORDER BY timestamp DESC 
                        LIMIT ? OFFSET ?
                    """, (cutoff_timestamp, downsample_factor, historical_limit, historical_offset))
                else:
                    cursor.execute("""
                        SELECT * FROM points 
                        WHERE timestamp < ? 
                        ORDER BY timestamp DESC 
                        LIMIT ? OFFSET ?
                    """, (cutoff_timestamp, historical_limit, historical_offset))
                
                historical_points = [dict(row) for row in cursor.fetchall()]
                print(f"Retrieved {len(historical_points)} historical points with limit={historical_limit}, offset={historical_offset}")
            except Exception as e:
                print(f"Error fetching historical data: {e}")
                historical_points = []
            
            # Process the data (convert JSON strings to objects)
            for point in current_points + historical_points:
                try:
                    if 'cpu_usage_per_core' in point and point['cpu_usage_per_core']:
                        point['cpu_usage_per_core'] = json.loads(point['cpu_usage_per_core'])
                    if 'gpu_usage' in point and point['gpu_usage']:
                        point['gpu_usage'] = json.loads(point['gpu_usage'])
                except Exception as e:
                    print(f"Error processing point data: {e}")
            
            # Return the full structure
            return {
                'current': current_points,
                'historical': historical_points,
                'meta': {
                    'requested': historical_limit,
                    'returned': len(historical_points),
                    'downsampled': downsample_factor > 1,
                    'factor': downsample_factor
                }
            }
    except Exception as e:
        print(f"Error in produce_json_structure: {e}")
        # Return empty data rather than causing error
        return {
            'current': [],
            'historical': []
        }

if __name__ == "__main__":
    # Skip data collection outside active hours
    if not is_active_hours():
        log("\n=== Outside active hours (8am-10pm), skipping data collection ===")
        sys.exit(0)
        
    # Example usage: run this script to do one minute of collection & save to DB.
    log("\n=== Script Started ===")
    init_db()
    minute_data = collect_minute_data()
    count_inserted = save_points_to_db(minute_data)
    data_struct = produce_json_structure()
    log(f"Just inserted {count_inserted} points. Total in DB: {len(data_struct['historical'])}")
    log("\n=== Script Completed ===\n")