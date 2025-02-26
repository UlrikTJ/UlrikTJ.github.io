#!/usr/bin/env python3

import os
import psutil
import json
import subprocess
import time
from datetime import datetime
import pytz
import sys

# Redirect all print statements to stderr so they never pollute data.json
def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Define the timezone
copenhagen_tz = pytz.timezone('Europe/Copenhagen')

def get_system_info():
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
    data_points = []
    start_time = time.time()
    log("\n=== Starting Data Collection ===")

    try:
        while time.time() - start_time < 60:
            current_time = datetime.now(copenhagen_tz).isoformat()
            info = get_system_info()
            point = {
                "timestamp": current_time,
                **info
            }
            data_points.append(point)
            
            # Update status on stderr
            log(f"\rCollecting... Points: {len(data_points):2d} | "
                f"CPU: {info['cpu_usage']:4.1f}% | "
                f"MEM: {info['memory_usage']:4.1f}%", end="")
            
            time.sleep(1)

        log("\nCollection complete!")
    except Exception as e:
        log(f"\nError during collection: {e}")

    return data_points

def save_data(new_points):
    log("\n=== Starting Save Operation ===")

    all_points = []

    # 1. Load existing data if any
    if os.path.exists('data.json'):
        # Read the file and handle empty content as "{}"
        try:
            with open('data.json', 'r') as f:
                file_content = f.read().strip()
                if file_content == "":
                    # It's empty, so treat as empty dict
                    existing_data = {}
                else:
                    existing_data = json.loads(file_content)

            if 'historical' in existing_data:
                all_points.extend(existing_data['historical'])
                log(f"Loaded {len(existing_data['historical'])} existing points")

        except json.JSONDecodeError as e:
            # If it's not empty but invalid JSON, abort
            log(f"Error loading existing data: {e}")
            log("Aborting to avoid overwriting old data.")
            return None
        except Exception as e:
            # If any other error, also abort
            log(f"Error loading existing data: {e}")
            log("Aborting to avoid overwriting old data.")
            return None

    # 2. Add new points
    all_points.extend(new_points)
    log(f"Added {len(new_points)} new points")

    # 3. Deduplicate by timestamp
    points_dict = {}
    for point in all_points:
        points_dict[point['timestamp']] = point

    # 4. Sort
    unique_points = list(points_dict.values())
    unique_points.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))

    log(f"Total unique points: {len(unique_points)}")

    # 5. Construct new data
    new_data = {
        'current': new_points,
        'historical': unique_points
    }

    # 6. Write to temp file, then replace
    temp_file = 'data.json.tmp'
    try:
        with open(temp_file, 'w') as f:
            json.dump(new_data, f, indent=4)
        
        # Verify
        with open(temp_file, 'r') as f:
            verify_data = json.load(f)
            if len(verify_data['historical']) != len(unique_points):
                raise ValueError("Verification failed: mismatch in historical points.")

        os.replace(temp_file, 'data.json')

        log("\nSave successful:")
        log(f"- Historical points: {len(unique_points)}")
        if len(unique_points) > 0:
            log(f"- Time range: {unique_points[0]['timestamp']} "
                f"to {unique_points[-1]['timestamp']}")
        log(f"- File size: {os.path.getsize('data.json')/1024:.2f}KB")

    except Exception as e:
        log(f"Error during save: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

    return len(unique_points)

def verify_file():
    try:
        with open('data.json', 'r') as f:
            file_content = f.read().strip()
            if not file_content:
                log("Verification error: data.json is empty.")
                return 0

            data = json.loads(file_content)
            hist_len = len(data.get('historical', []))
            curr_len = len(data.get('current', []))
            
            if hist_len > 0:
                first = data['historical'][0]['timestamp']
                last = data['historical'][-1]['timestamp']
                log(f"\nFile verification:")
                log(f"- Historical points: {hist_len}")
                log(f"- Current points: {curr_len}")
                log(f"- Time range: {first} to {last}")
            return hist_len
    except Exception as e:
        log(f"Verification error: {e}")
        return 0

if __name__ == "__main__":
    log("\n=== Script Started ===")

    # Check existing data
    initial_count = verify_file()

    # Collect data for one minute
    minute_data = collect_minute_data()

    if minute_data:
        final_count = save_data(minute_data)

        # Re-verify if save_data succeeded
        if final_count is not None:
            verify_file()
            log(f"\nPoints growth: {initial_count} -> {final_count}")

    log("\n=== Script Completed ===")