#!/bin/bash

# Get current hour in Copenhagen time
HOUR=$(TZ="Europe/Copenhagen" date +%H)

# Check if current time is within active hours (8-22)
if [ "$HOUR" -ge 0 ] && [ "$HOUR" -lt 23.95 ]; then
    echo "Starting Flask server (active hours: 8am-10pm Copenhagen time)"
    exec /home/mainserver/myflask_venv/bin/python /home/mainserver/UlrikTJ.github.io/server.py
else
    echo "Outside active hours (8am-10pm Copenhagen time), exiting with success status"
    exit 0  # Exit with success status so systemd doesn't consider it failed
fi
