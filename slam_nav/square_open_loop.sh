#!/usr/bin/env bash
set -u  # no set -e (important!)

TOPIC="/cmd_vel"
RATE=10

sleep 3  # wait for ROS2 to be ready

# Your current values
FWD_T="12"
TURN_T="6.5"
V="0.1"
W="-0.3"     # negative = turn right
STOP_T="0.3"

run_twist () {
  local dur="$1"
  local vx="$2"
  local wz="$3"

  # timeout returns 124 when it stops the command; treat that as success
  timeout "${dur}s" ros2 topic pub -r "${RATE}" "${TOPIC}" geometry_msgs/msg/Twist \
    "{linear: {x: ${vx}}, angular: {z: ${wz}}}" >/dev/null 2>&1
  local ec=$?
  if [[ $ec -ne 0 && $ec -ne 124 ]]; then
    echo "Command failed (exit code $ec)"
    exit $ec
  fi
}

stop_robot () {
  # publish a couple of zero cmds to be safe
  run_twist "0.2" "0.0" "0.0"
  ros2 topic pub --once "${TOPIC}" geometry_msgs/msg/Twist \
    "{linear: {x: 0.0}, angular: {z: 0.0}}" >/dev/null 2>&1 || true
}

trap stop_robot EXIT INT TERM

echo "Square open-loop: forward ${FWD_T}s @ ${V}, turn ${TURN_T}s @ ${W}, rate ${RATE}Hz"

for i in 1 2 3 4; do
  echo "Side $i: forward"
  run_twist "${FWD_T}" "${V}" "0.0"
  run_twist "${STOP_T}" "0.0" "0.0"

  echo "Corner $i: turn"
  run_twist "${TURN_T}" "0.0" "${W}"
  run_twist "${STOP_T}" "0.0" "0.0"
done

echo "Done."
