#!/bin/bash

# Check if ethtool is installed
if ! dpkg -l | grep -q "ethtool"; then
    echo "\e[31mError: ethtool is not installed on this system.\e[0m"
    echo "Please install it using:"
    echo "sudo apt update && sudo apt install ethtool"
    exit 1
fi

# Check if can-utils is installed
if ! dpkg -l | grep -q "can-utils"; then
    echo "\e[31mError: can-utils is not installed on this system.\e[0m"
    echo "Please install it using:"
    echo "sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "Both ethtool and can-utils are installed."

# Iterate through all CAN interfaces
for iface in $(ip -br link show type can | awk '{print $1}'); do
    # Use ethtool to get bus-info
    BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
    
    if [ -z "$BUS_INFO" ]; then
        echo "Error: Could not retrieve bus-info for interface $iface."
        continue
    fi
    
    echo "Interface $iface is connected at USB port $BUS_INFO"
done
