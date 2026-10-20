#!/bin/bash

<<!EOF!
查看节点占用情况的小工具
作者：董建艇，cursor
修改于：2026.9.7
!EOF!

nodes=(20 21 22 23 38 39 40 41 43 48 49 50)
result_dir=$(mktemp -d "${TMPDIR:-/tmp}/tops.XXXXXX")

cleanup() {
    rm -rf "$result_dir"
}
trap cleanup EXIT

printf "\n"
# print a big "HUST-SPIN"
cat <<'EOF'
                    ██╗   ██╗██╗   ██╗████████╗████████╗         ████████╗███████╗ ████████╗███╗  ██╗
                    ██║   ██║██║   ██║██╔═════╝╚══██╔══╝         ██╔═════╝██╔═══██╗╚══██╔══╝████╗ ██║
                    ████████║██║   ██║████████╗   ██║   ████████╗████████╗███████╔╝   ██║   ██╔██╗██║
                    ██╔═══██║██║   ██║╚═════██║   ██║   ╚═══════╝╚═════██║██╔════╝    ██║   ██║╚████║
                    ██║   ██║╚██████╔╝████████║   ██║            ████████║██║      ████████╗██║ ╚███║
                    ╚═╝   ╚═╝ ╚═════╝ ╚═══════╝   ╚═╝            ╚═══════╝╚═╝      ╚═══════╝╚═╝  ╚══╝

Node|    PowerOn|CpuUsed|TaskNum|FreeMem|UsedSwap|               Users               |TempCpu1|TempCpu2|Temphigh|TempCrit
EOF

collect_node() {
    local node="$1"
    local output_file="$result_dir/cu${node}.raw"

    if ! ssh -q \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -o ConnectionAttempts=1 \
        -o ServerAliveInterval=3 \
        -o ServerAliveCountMax=1 \
        -o StrictHostKeyChecking=no \
        "cu${node}" '
            printf "__TOP_BEGIN__\n"
            top -b -n 1 | sed -n "1,178p"
            printf "__TOP_END__\n"
            printf "__SENSORS_BEGIN__\n"
            sensors
            printf "__SENSORS_END__\n"
        ' >"$output_file" 2>/dev/null; then
        printf "OFFLINE\n" >"$output_file"
    fi
}

# Each node is independent, so collect all nodes concurrently.
for node in "${nodes[@]}"; do
    collect_node "$node" &
done
wait

for nodes in "${nodes[@]}"; do
    output_file="$result_dir/cu${nodes}.raw"

    if [[ ! -s "$output_file" ]] || grep -q '^OFFLINE$' "$output_file"; then
        printf "Warning! cu%-3s Not online, please check\n" "${nodes}"
        continue
    fi

    find=$(sed -n '/^__TOP_BEGIN__$/,/^__TOP_END__$/p' "$output_file" | sed '1d;$d')
    temperature_data=$(sed -n '/^__SENSORS_BEGIN__$/,/^__SENSORS_END__$/p' "$output_file" | sed '1d;$d')

    # Parse the top output locally, avoiding one process per field.
    read -r poweron tasks cpu freemem usedswap < <(
        awk '
            NR == 1 { poweron = $5 }
            NR == 2 { tasks = $4 }
            NR == 3 { cpu = $2 }
            NR == 4 { freemem = $6 }
            NR == 5 { usedswap = $7 }
            END { print poweron, tasks, cpu, freemem, usedswap }
        ' <<< "$find"
    )

    # Transfer the unit to Gb.
    freemem=$(sed -n '4p' <<< "$find" | sed 's/+/0 /g' | awk '{printf $6}')
    usedswap=$(sed -n '5p' <<< "$find" | sed s/+/'0 '/g | awk '{printf $7}')
    freemem=${freemem%%.*}
    usedswap=${usedswap%%.*}
    freemem=${freemem:-0}
    usedswap=${usedswap:-0}
    freemem=$((freemem / 1048))
    usedswap=$((usedswap / 1048))

    # Know who uses this node, divided by 24 cores, i.e. 24 lines.
    user1=$(sed -n '8p' <<< "$find" | awk '{printf $2}')
    user2=$(sed -n '32p' <<< "$find" | awk '{printf $2}')
    user3=$(sed -n '56p' <<< "$find" | awk '{printf $2}')
    user4=$(sed -n '80p' <<< "$find" | awk '{printf $2}')
    user5=$(sed -n '104p' <<< "$find" | awk '{printf $2}')
    user6=$(sed -n '128p' <<< "$find" | awk '{printf $2}')
    user7=$(sed -n '152p' <<< "$find" | awk '{printf $2}')
    user8=$(sed -n '176p' <<< "$find" | awk '{printf $2}')

    case $nodes in
      20|21|22|23)
        temperature_info=$(grep 'Package' <<< "$temperature_data")
        node_users=$(printf "%8s" "${user1}")
        ;;
      38|39|40|41)
        temperature_info=$(grep 'Package' <<< "$temperature_data")
        node_users=$(printf "%8s %8s" "${user1}" "${user2}")
        ;;
      43)
        temperature_info=$(grep 'Package' <<< "$temperature_data")
        node_users=$(printf "%8s %8s %8s %8s" "${user1}" "${user2}" "${user3}" "${user4}")
        ;;
      48|49|50)
        temperature_info=$(grep 'Tctl' <<< "$temperature_data")
        node_users=$(printf "%8s %8s %8s %8s" "${user1}" "${user3}" "${user5}" "${user7}")
        ;;
    esac

    temperature_cpu1=$(sed -n '1p' <<< "$temperature_info" | grep -oE '\+[0-9]{2,3}' | xargs)
    temperature_cpu2=$(sed -n '2p' <<< "$temperature_info" | grep -oE '\+[0-9]{2,3}' | xargs)
    temperature_cpu1_now=$(awk '{printf $1}' <<< "$temperature_cpu1")
    temperature_cpu2_now=$(awk '{printf $1}' <<< "$temperature_cpu2")
    temperature_high=$(awk '{printf $2}' <<< "$temperature_cpu1")
    temperature_crit=$(awk '{printf $3}' <<< "$temperature_cpu1")

    case $nodes in
      20|21|22|23|38|39|40|41|43)
        temp_fields=$(printf "%8s|%8s|%8s|%8s" "${temperature_cpu1_now}C" "${temperature_cpu2_now}C" "${temperature_high}C" "${temperature_crit}C")
        printf "cu%2s|%11s|%7s|%7s|%7s|%8s|%35s|%8s\n" "${nodes}" "${poweron} days" "${cpu}%" "${tasks}" "${freemem}G" "${usedswap}G" "${node_users}" "${temp_fields}"
        ;;
      48|49|50)
        temp_fields=$(printf "%8s|%8s" "${temperature_cpu1_now}C" "${temperature_cpu2_now}C")
        printf "cu%2s|%11s|%7s|%7s|%7s|%8s|%35s|%8s\n" "${nodes}" "${poweron} days" "${cpu}%" "${tasks}" "${freemem}G" "${usedswap}G" "${node_users}" "${temp_fields}"
        ;;
    esac
done

printf "\n"
