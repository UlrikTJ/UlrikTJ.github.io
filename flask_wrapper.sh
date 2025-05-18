#!/bin/bash

# Get current time in Copenhagen timezone (hour and minute)
HOUR=$(TZ="Europe/Copenhagen" date +%H)
MINUTE=$(TZ="Europe/Copenhagen" date +%M)

# Convert to total minutes for easier comparison
CURRENT_TIME_MINUTES=$((HOUR * 60 + MINUTE))

# Define maintenance window (e.g., 3:00-3:05 AM = 180-185 minutes after midnight)
MAINTENANCE_START=180  # 3:00 AM in minutes
MAINTENANCE_END=185    # 3:05 AM in minutes

# Check if current time is within the maintenance window
if [ "$CURRENT_TIME_MINUTES" -ge "$MAINTENANCE_START" ] && [ "$CURRENT_TIME_MINUTES" -lt "$MAINTENANCE_END" ]; then
    echo "Maintenance window (3:00-3:05 AM Copenhagen time), server not starting"
    exit 0  # Exit with success status so systemd doesn't consider it failed
else
    echo "Starting Flask server (running 24/7 except for 5-minute daily maintenance)"
    exec /home/mainserver/myflask_venv/bin/python /home/mainserver/UlrikTJ.github.io/server.py
fi