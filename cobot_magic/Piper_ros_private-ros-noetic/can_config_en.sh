#!/bin/bash

# Usage instructions:
#
# 1. Prerequisites
#     Install ip tool and ethtool
#     sudo apt install ethtool can-utils
#     Make sure the gs_usb driver is properly loaded/installed.
#
# 2. Purpose
#     This script automatically detects, renames and brings up CAN interfaces
#     based on their USB port location.
#     Especially useful when you have multiple CAN adapters and want stable/persistent names.
#
# 3. Main features
#     - Checks that the number of detected CAN devices matches expectation
#     - Reads USB port (bus-info) of each CAN interface using ethtool
#     - Renames interface according to predefined USB port → name mapping
#     - Sets bitrate and activates the interface
#
# 4. Configuration
#     Key settings:
#     1. Expected number of CAN adapters
#        EXPECTED_CAN_COUNT=3
#     2. For single CAN adapter – default name and bitrate
#        (can be overridden via command-line arguments)
#     3. For multiple CAN adapters – USB port to name+bitrate mapping
#        declare -A USB_PORTS
#        USB_PORTS["1-13:1.0"]="can_left:1000000"
#        USB_PORTS["1-3:1.0"]="can_right:1000000"
#        USB_PORTS["1-2:1.0"]="can_base:500000"
#
# 5. How to prepare (especially for multiple adapters)
#     1. Insert one adapter at a time into the desired USB port
#     2. Run:   sudo ethtool -i canX | grep bus
#        Record the bus-info value (e.g. 1-3:1.0)
#     3. Repeat for each adapter / port
#     4. Update the USB_PORTS array with the recorded bus-info values
#
# 6. Running the script
#     Single adapter mode:
#       sudo ./can_config.sh           → uses can0 + 1000000
#       sudo ./can_config.sh can_mine 500000
#       sudo ./can_config.sh vcan_left 1000000 1-3:1.0
#
#     Multiple adapters mode:
#       Just run (no arguments needed):
#       sudo ./can_config.sh
#
#-------------------------------------------------------------------------------------------------#

# Expected number of CAN adapters
EXPECTED_CAN_COUNT=3

# ── Single adapter mode variables ───────────────────────────────────────
if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    # Default interface name (can be overridden by $1)
    DEFAULT_CAN_NAME="${1:-can0}"

    # Default bitrate in bit/s (can be overridden by $2)
    DEFAULT_BITRATE="${2:-1000000}"

    # Optional: specific USB bus-info to match (can be overridden by $3)
    USB_ADDRESS="${3}"
fi

# ── Multiple adapters mode configuration ────────────────────────────────
if [ "$EXPECTED_CAN_COUNT" -ne 1 ]; then
    declare -A USB_PORTS
    USB_PORTS["1-12:1.0"]="can_left:1000000"
    USB_PORTS["1-3:1.0"]="can_right:1000000"
    USB_PORTS["1-2:1.0"]="can_0:500000"
fi

# Count how many CAN interfaces are currently visible
CURRENT_CAN_COUNT=$(ip link show type can | grep -c "link/can")

# Safety check: number of devices
if [ "$CURRENT_CAN_COUNT" -ne "$EXPECTED_CAN_COUNT" ]; then
    echo "Error: Detected $CURRENT_CAN_COUNT CAN interfaces, but expected $EXPECTED_CAN_COUNT."
    exit 1
fi

# Make sure the gs_usb kernel module is loaded
sudo modprobe gs_usb
if [ $? -ne 0 ]; then
    echo "Error: Failed to load gs_usb module."
    exit 1
fi

# ── Single CAN adapter handling ─────────────────────────────────────────
if [ "$EXPECTED_CAN_COUNT" -eq 1 ]; then
    if [ -n "$USB_ADDRESS" ]; then
        echo "USB address filter provided: $USB_ADDRESS"

        INTERFACE_NAME=""
        for iface in $(ip -br link show type can | awk '{print $1}'); do
            BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
            if [ "$BUS_INFO" = "$USB_ADDRESS" ]; then
                INTERFACE_NAME="$iface"
                break
            fi
        done

        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: No CAN interface found with bus-info $USB_ADDRESS"
            exit 1
        else
            echo "Found matching interface: $INTERFACE_NAME"
        fi
    else
        # Just take the only CAN interface
        INTERFACE_NAME=$(ip -br link show type can | awk '{print $1}')

        if [ -z "$INTERFACE_NAME" ]; then
            echo "Error: No CAN interface detected."
            exit 1
        fi

        echo "Single CAN expected → found interface: $INTERFACE_NAME"
    fi

    # Check if already up
    IS_LINK_UP=$(ip link show "$INTERFACE_NAME" | grep -q "UP" && echo "yes" || echo "no")

    # Get current bitrate (if set)
    CURRENT_BITRATE=$(ip -details link show "$INTERFACE_NAME" | grep -oP 'bitrate \K\d+' || echo "0")

    if [ "$IS_LINK_UP" = "yes" ] && [ "$CURRENT_BITRATE" -eq "$DEFAULT_BITRATE" ]; then
        echo "Interface $INTERFACE_NAME is already up @ $DEFAULT_BITRATE bit/s"

        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming $INTERFACE_NAME → $DEFAULT_CAN_NAME"
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
            echo "Rename complete."
        else
            echo "Name is already correct ($DEFAULT_CAN_NAME)"
        fi
    else
        if [ "$IS_LINK_UP" = "yes" ]; then
            echo "Interface is up but bitrate is $CURRENT_BITRATE (wanted $DEFAULT_BITRATE)"
        else
            echo "Interface is down or bitrate not set."
        fi

        echo "Configuring $INTERFACE_NAME → $DEFAULT_BITRATE bit/s"
        sudo ip link set "$INTERFACE_NAME" down
        sudo ip link set "$INTERFACE_NAME" type can bitrate "$DEFAULT_BITRATE"
        sudo ip link set "$INTERFACE_NAME" up

        if [ "$INTERFACE_NAME" != "$DEFAULT_CAN_NAME" ]; then
            echo "Renaming $INTERFACE_NAME → $DEFAULT_CAN_NAME"
            sudo ip link set "$INTERFACE_NAME" down
            sudo ip link set "$INTERFACE_NAME" name "$DEFAULT_CAN_NAME"
            sudo ip link set "$DEFAULT_CAN_NAME" up
        fi

        echo "Setup complete for $DEFAULT_CAN_NAME"
    fi
else
    # ── Multiple CAN adapters handling ──────────────────────────────────

    PREDEFINED_COUNT=${#USB_PORTS[@]}
    if [ "$EXPECTED_CAN_COUNT" -ne "$PREDEFINED_COUNT" ]; then
        echo "Error: Expected $EXPECTED_CAN_COUNT devices, but only ${PREDEFINED_COUNT} port mappings defined."
        exit 1
    fi

    for iface in $(ip -br link show type can | awk '{print $1}'); do
        BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')

        if [ -z "$BUS_INFO" ]; then
            echo "Error: Cannot read bus-info for interface $iface"
            continue
        fi

        echo "Interface $iface is connected at USB port: $BUS_INFO"

        if [ -n "${USB_PORTS[$BUS_INFO]}" ]; then
            IFS=':' read -r TARGET_NAME TARGET_BITRATE <<< "${USB_PORTS[$BUS_INFO]}"

            IS_LINK_UP=$(ip link show "$iface" | grep -q "UP" && echo "yes" || echo "no")
            CURRENT_BITRATE=$(ip -details link show "$iface" | grep -oP 'bitrate \K\d+' || echo "0")

            if [ "$IS_LINK_UP" = "yes" ] && [ "$CURRENT_BITRATE" -eq "$TARGET_BITRATE" ]; then
                echo "  → already up @ $TARGET_BITRATE bit/s"

                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "  → renaming to $TARGET_NAME"
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                else
                    echo "  → name already correct"
                fi
            else
                if [ "$IS_LINK_UP" = "yes" ]; then
                    echo "  → up but wrong bitrate ($CURRENT_BITRATE ≠ $TARGET_BITRATE)"
                else
                    echo "  → down or bitrate not set"
                fi

                echo "  → setting up $TARGET_NAME @ $TARGET_BITRATE bit/s"
                sudo ip link set "$iface" down
                sudo ip link set "$iface" type can bitrate "$TARGET_BITRATE"
                sudo ip link set "$iface" up

                if [ "$iface" != "$TARGET_NAME" ]; then
                    echo "  → renaming to $TARGET_NAME"
                    sudo ip link set "$iface" down
                    sudo ip link set "$iface" name "$TARGET_NAME"
                    sudo ip link set "$TARGET_NAME" up
                fi
            fi
        else
            echo "Error: Unknown USB port $BUS_INFO (interface $iface)"
            exit 1
        fi
    done
fi

echo "All CAN interfaces have been configured, renamed and brought up successfully."