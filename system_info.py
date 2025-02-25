import os
import psutil
import json
import subprocess
import time
from datetime import datetime, timedelta
import pytz
import sys

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
                "usage": int(usage.replace(" %", ""))  # Remove '%' and convert to integer
            })
        return gpu_info
    except Exception as e:
        print("Error fetching GPU usage:", e)  # Debug print
        return None

# At the top of your script, define the timezone
copenhagen_tz = pytz.timezone('Europe/Copenhagen')

def debug_print(message, end='\n'):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}", end=end, flush=True)

def collect_minute_data():
    data_points = []
    start_time = time.time()
    print("\n=== Starting Data Collection ===")
    sys.stdout.flush()
    
    try:
        while time.time() - start_time < 60:
            current_time = datetime.now(copenhagen_tz).isoformat()
            info = get_system_info()
            point = {
                "timestamp": current_time,
                **info
            }
            data_points.append(point)
            
            # Clear line and update status
            sys.stdout.write('\r' + ' ' * 80)  # Clear the line
            sys.stdout.write(f"\rCollecting... Points: {len(data_points):2d} | CPU: {info['cpu_usage']:4.1f}% | MEM: {info['memory_usage']:4.1f}%")
            sys.stdout.flush()
            
            time.sleep(1)
        
        print("\nCollection complete!")
        sys.stdout.flush()
    except Exception as e:
        print(f"\nError during collection: {e}")
        sys.stdout.flush()
    
    return data_points

def save_data(data_points):
    print("\n=== Starting Save Operation ===")
    sys.stdout.flush()
    
    try:
        # Load existing data or create new structure
        if os.path.exists('data.json') and os.path.getsize('data.json') > 0:
            with open('data.json', 'r') as f:
                stored_data = json.load(f)
                print(f"Loaded existing data with {len(stored_data['historical'])} historical points")
        else:
            stored_data = {"current": [], "historical": []}
            print("No existing data found, creating new structure")
        
        # Update current data
        stored_data["current"] = data_points
        
        # Append new points to historical data
        stored_data["historical"].extend(data_points)
        print(f"Added {len(data_points)} new points")
        
        # Remove duplicates based on timestamp
        unique_points = {}
        for point in stored_data["historical"]:
            unique_points[point["timestamp"]] = point
        stored_data["historical"] = list(unique_points.values())
        
        # Sort historical data by timestamp
        stored_data["historical"].sort(
            key=lambda x: datetime.fromisoformat(x["timestamp"])
        )
        
        # Keep only last year of data
        current_time = datetime.now(copenhagen_tz)
        one_year_ago = current_time - timedelta(days=365)
        
        stored_data["historical"] = [
            point for point in stored_data["historical"]
            if datetime.fromisoformat(point["timestamp"]) > one_year_ago
        ]
        
        print(f"Total historical points after cleanup: {len(stored_data['historical'])}")
        
        # Save the updated data
        with open('data.json.tmp', 'w') as f:
            json.dump(stored_data, f, indent=4)
        
        # Replace the original file
        os.replace('data.json.tmp', 'data.json')
        
        file_size = os.path.getsize('data.json')
        print(f"Save successful - File size: {file_size/1024:.2f}KB")
        
    except Exception as e:
        print(f"Error during save operation: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== Save Operation Complete ===\n")
    sys.stdout.flush()

def verify_data():
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
            
        timestamps = set()
        for point in data["historical"]:
            timestamps.add(point["timestamp"])
        
        print("\n=== Data Verification ===")
        print(f"Current points: {len(data['current'])}")
        print(f"Historical points: {len(data['historical'])}")
        print(f"Unique timestamps: {len(timestamps)}")
        
        # Show time range
        if data["historical"]:
            first_time = min(datetime.fromisoformat(p["timestamp"]) for p in data["historical"])
            last_time = max(datetime.fromisoformat(p["timestamp"]) for p in data["historical"])
            print(f"Time range: {first_time} to {last_time}")
            print(f"Duration: {last_time - first_time}")
        
    except Exception as e:
        print(f"Verification error: {e}")

if __name__ == "__main__":
    print("\n=== Script Started ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"User ID: {os.getuid()}")
    print(f"Directory permissions: {oct(os.stat('.').st_mode)[-3:]}")
    sys.stdout.flush()
    
    minute_data = collect_minute_data()
    if minute_data:
        save_data(minute_data)
        verify_data()
    else:
        print("No data collected!")
    
    print("=== Script Completed ===\n")
    sys.stdout.flush()