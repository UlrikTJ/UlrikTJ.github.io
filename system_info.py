import psutil
import json

def get_system_info():
    info = {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "cpu_temp": get_cpu_temp(),
        "uptime": psutil.boot_time()
    }
    return info

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if 'coretemp' in temps:
            return temps['coretemp'][0].current
        return None
    except:
        return None

if __name__ == "__main__":
    print(json.dumps(get_system_info()))
