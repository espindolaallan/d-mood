#!/bin/bash
# This script can be used to call adjust_pcap_timestamps.py, payload_manipulation.py and time_volume_manipulations.py
# List of dates to process
dates=("22-1-2015" "17-2-2015")

script="volume"

#factors=("-90" "-80" "-70" "-60" "-50" "-40" "-30" "-20" "-10" "10" "20" "30" "40" "50" "60" "70" "80" "90" "100")
factors=("-80" "-70")
for factor in "${factors[@]}"
do
    echo "Working with factor: ${factor}"

    for date in "${dates[@]}"
    do
        echo "Processing date: ${date}"

        # base pcaps
        input_path="../../UNSW-NB15-pcap_files/pcaps_${date}/"
        #input_path="../../UNSW-NB15-delay/UNSW-NB15-pcap_files_no_payload/pcaps_${date}/"

        if [ "$script" = "time" ]; then
            echo "Crafting Time Attacks"
            output_path="../../UNSW-NB15-Attacks/UNSW-NB15-pcap_files_${factor}xtimedelay/pcaps_${date}/"
            python3 ../EspPipeML/adjust_pcap_timestamps.py "${input_path}" "${output_path}" --factor ${factor}

        elif [ "$script" = "volume" ]; then
            echo "Crafting Volume Attacks"
            output_path="../../UNSW-NB15-Attacks/UNSW-NB15-pcap_files_${factor}percent_payload/pcaps_${date}/"
            python3 ../EspPipeML/payload_manipulation.py "${input_path}" "${output_path}" ${factor}

        elif [ "$script" = "time_volume" ]; then
            echo "Crafting Time & Volume Attacks"
            output_path="../../UNSW-NB15-Attacks/UNSW-NB15-pcap_files_${factor}_time_volume/pcaps_${date}/"
            python3 ../EspPipeML/time_volume_manipulations.py "${input_path}" "${output_path}" ${factor}
        fi

    done
done