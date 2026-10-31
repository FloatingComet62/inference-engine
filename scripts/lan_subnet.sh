#!/usr/bin/env bash
# Prints the LAN subnet of this machine's default route (e.g. 192.168.1.0/24), for
# Open MPI's *_if_include: a subnet works on every machine whatever its interface is called.
dev=$(ip -o -4 route show default | awk '{for (i = 1; i < NF; i++) if ($i == "dev") { print $(i + 1); exit }}')
cidr=$(ip -o -4 addr show dev "$dev" | awk '{print $4; exit}')
python3 -c 'import ipaddress, sys; print(ipaddress.ip_interface(sys.argv[1]).network)' "$cidr"
