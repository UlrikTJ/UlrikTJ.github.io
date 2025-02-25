import psutil
import json
import subprocess
import time
from datetime import datetime, timedelta
import pytz

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

# Then modify where you create the timestamp
def collect_minute_data():
    data_points = []
    start_time = time.time()
    
    while time.time() - start_time < 60:
        # Get current time in Copenhagen timezone
        current_time = datetime.now(copenhagen_tz).isoformat()
        data_points.append({
            "timestamp": current_time,
            **get_system_info()
        })
        time.sleep(1)
    
    return data_points

def save_data(data_points):
    try:
        with open('data.json', 'r') as f:
            stored_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        stored_data = {"current": [], "historical": []}
    
    # Update current data
    stored_data["current"] = data_points
    
    # Add new points to historical data
    stored_data["historical"].extend(data_points)
    
    # Keep only last year of data
    current_time = datetime.now(copenhagen_tz)
    one_year_ago = current_time - timedelta(days=365)
    
    # Filter and sort historical data
    stored_data["historical"] = sorted(
        [
            point for point in stored_data["historical"]
            if datetime.fromisoformat(point["timestamp"].split('+')[0]) > one_year_ago
        ],
        key=lambda x: datetime.fromisoformat(x["timestamp"].split('+')[0])
    )
    
    # Optional: Limit the number of data points
    max_points = 100000
    if len(stored_data["historical"]) > max_points:
        stored_data["historical"] = stored_data["historical"][-max_points:]
    
    # Save the data
    with open('data.json', 'w') as f:
        json.dump(stored_data, f, indent=4)
    
if __name__ == "__main__":
    minute_data = collect_minute_data()
    save_data(minute_data)
    
