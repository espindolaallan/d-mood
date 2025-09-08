#!/bin/bash

#This script must be called inside go-flows docker!

# An array of dates for which the script will be run
declare -a dates=("22-1-2015" "17-2-2015")

# Base paths, adjust as needed
base_input_path_prefix="/mnt/pcaps/UNSW-NB15-pcap_files_"
base_save_path_prefix="/mnt/datasets/UNSW-NB15/go-flows/csv_files/csv_"

# Path to the python script
python_run_go_flows="/mnt/datasets/UNSW-NB15/go-flows/run_go-flows.py"

# Iterating over the slow parameter from 1.1 to 1.9 with step 0.1
#for slow in $(seq 1.1 0.1 1.9); do
#declare -a files=("10" "20" "30" "40" "50" "60" "100" "-10" "-20" "-30" "-40" "-50" "-60" "-90")
#declare -a files=("0.2" "0.5" "0.7" "1.1" "1.2" "1.3" "2" "3" "4" "50" "100")
#declare -a files=("0.7" "1.1" "1.2" "1.3" "2" "3" "4" "50" "100")
#declare -a files=("-90" "-80" "-70" "-60" "-50" "-40" "-30" "-20" "-10")
declare -a files=("-80" "-70")
attack_type="volume"

for factor in "${files[@]}"; do

    # Iterate over each date
    for date in "${dates[@]}"; do
        # Extracting only the day from the date string
        day=$(echo "$date" | cut -d- -f1)

        # Define the input and output paths
        if [ "$attack_type" = "time" ]; then
            echo "Export Time Attacks"
            
            input_path="${base_input_path_prefix}${factor}xtimedelay/pcaps_${date}/"
            save_path="${base_save_path_prefix}${factor}x_slow/day_${day}"

        elif [ "$attack_type" = "volume" ]; then
            echo "Export Volume Attacks"
            input_path="${base_input_path_prefix}${factor}percent_payload/pcaps_${date}/"
            save_path="${base_save_path_prefix}${factor}percent_payload/day_${day}"

        elif [ "$attack_type" = "time_volume" ]; then
            echo "Export Time & Volume Attacks"
            input_path="${base_input_path_prefix}${factor}_time_volume/pcaps_${date}/"
            save_path="${base_save_path_prefix}${factor}_time_volume/day_${day}"
        fi
        
        # Print information about the current iteration
        echo "------------------------------------"
        echo "Processing for date: $date"
        echo "Using factor parameter: $factor"
        echo "Input path: $input_path"
        echo "Save path: $save_path"
        echo "------------------------------------"
        
        # Running the Python script with the constructed paths
        python3 $python_run_go_flows "$input_path" "$save_path"
    done
done
