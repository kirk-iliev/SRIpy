#!/usr/bin/env bash
# Launch SRIpy with the right environment.
# Used by the desktop shortcut; can also be run from a terminal.

SRIPY_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
CAMERA_IP="10.16.18.50"
LOG_DIR="$HOME/.local/state/sripy"
LOG_FILE="$LOG_DIR/sripy_$(date +%Y%m%d_%H%M%S).log"

have_gui() { [ -n "$DISPLAY$WAYLAND_DISPLAY" ] && command -v zenity >/dev/null; }

# Vimba X camera driver path (normally set at login; set here so the
# shortcut works even before the user has logged out/in).
if [ -z "$GENICAM_GENTL64_PATH" ]; then
    for f in /etc/profile.d/VimbaX_GenTL_Path_64bit.sh; do
        [ -r "$f" ] && . "$f"
    done
fi
if [ -z "$GENICAM_GENTL64_PATH" ]; then
    VIMBA_CTI="$(ls -d /opt/VimbaX_*/cti 2>/dev/null | sort | tail -1)"
    [ -n "$VIMBA_CTI" ] && export GENICAM_GENTL64_PATH="$VIMBA_CTI"
fi

# Camera check: if the Manta isn't reachable SRIpy silently opens a
# Vimba *simulator* camera instead, so warn the user up front.
if ! ping -c1 -W1 "$CAMERA_IP" >/dev/null 2>&1; then
    MSG="The camera is not reachable at $CAMERA_IP.

Check that:
  • the camera is powered and plugged into the laptop's Ethernet port
  • the port is on the \"iGp12-local\" network (not DCCT-Local)

If you continue, SRIpy will open a SIMULATED camera, not the real one."
    if have_gui; then
        zenity --question --title="SRIpy - camera not found" --width=420 \
               --ok-label="Start anyway" --cancel-label="Cancel" --text="$MSG" || exit 0
    else
        echo "WARNING: $MSG" >&2
    fi
fi

mkdir -p "$LOG_DIR"
# Keep only the 20 most recent logs
ls -1t "$LOG_DIR"/sripy_*.log 2>/dev/null | tail -n +21 | xargs -r rm -f

cd "$SRIPY_DIR" || exit 1
"$SRIPY_DIR/.venv/bin/python" main.py "$@" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}

if [ "$STATUS" -ne 0 ] && have_gui; then
    zenity --error --title="SRIpy exited with an error" --width=600 \
           --text="SRIpy stopped with exit code $STATUS.\n\nFull log: $LOG_FILE\n\nLast lines:\n$(tail -n 15 "$LOG_FILE" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')"
fi
exit "$STATUS"
