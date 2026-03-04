#!/bin/bash
cd "$(dirname "$0")"

echo "=== Run 1 ==="
python control_single_bus.py

echo "=== Run 2 ==="
python control_single_bus.py
