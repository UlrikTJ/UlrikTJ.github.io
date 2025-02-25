import psutil
import json
import subprocess
import time
from datetime import datetime

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

def collect_minute_data():
    data_points = []
    start_time = time.time()
    
    # Collect data for 60 seconds
    while time.time() - start_time < 60:
        data_points.append({
            "timestamp": datetime.now().isoformat(),
            **get_system_info()
        })
        time.sleep(1)  # Sample every second
    
    # Calculate averages for the minute
    minute_data = {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage": sum(d["cpu_usage"] for d in data_points) / len(data_points),
        "memory_usage": sum(d["memory_usage"] for d in data_points) / len(data_points),
        "disk_usage": sum(d["disk_usage"] for d in data_points) / len(data_points),
        "cpu_temp": sum(d["cpu_temp"] for d in data_points) / len(data_points) if data_points[0]["cpu_temp"] else None,
        "gpu_temp": sum(d["gpu_temp"] for d in data_points) / len(data_points) if data_points[0]["gpu_temp"] else None,
        "nvme_temp": sum(d["nvme_temp"] for d in data_points) / len(data_points) if data_points[0]["nvme_temp"] else None,
        "uptime": data_points[-1]["uptime"]
    }
    
    return minute_data

# Modify your main script to save historical data
if __name__ == "__main__":
    # Load existing historical data
    try:
        with open('historical_data.json', 'r') as f:
            historical_data = json.load(f)
    except FileNotFoundError:
        historical_data = []
    
    # Add new minute data
    minute_data = collect_minute_data()
    historical_data.append(minute_data)
    
    # Keep only last 24 hours of data (1440 minutes)
    if len(historical_data) > 1440:
        historical_data = historical_data[-1440:]
    
    # Save both current and historical data
    with open('data.json', 'w') as f:
        json.dump({"current": minute_data, "historical": historical_data}, f)
