import argparse
import time
import logging
import os
from multiprocessing import Pool
from scapy.all import *

# Initialize logging
logging.basicConfig(filename='adjust_timestamps.log', level=logging.INFO)

def create_dir_if_not_exists(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def adjust_timestamps(args):
    input_file, output_dir, factor = args
    total_start_time = time.time()

    # Create the output file path using the output directory and the name of the input file
    output_file = os.path.join(output_dir, os.path.basename(input_file))

    # Read packets from the original PCAP file
    packets = rdpcap(input_file)
    time_stamp_first_pkt = packets[0].time

    # Loop through each packet to modify the timestamp
    for i, packet in enumerate(packets):
        if i == 0:  # Skip the first packet
            continue
        #difference = time_stamp_first_pkt - packet.time
        difference = packet.time - time_stamp_first_pkt
        # Update the current packet's timestamp
        #updated_time_stamp_current_pkt = (difference * factor) + time_stamp_first_pkt
        updated_time_stamp_current_pkt = (difference * (1 + factor / 100)) + time_stamp_first_pkt
        packet.time = max(updated_time_stamp_current_pkt, time_stamp_first_pkt) # Prevent negative or non-sequential timestamp

    # Write packets to a new PCAP file
    wrpcap(output_file, packets)
    total_end_time = time.time()
    logging.info(f"Total time taken for {input_file}: {total_end_time - total_start_time} seconds.")

def main():
    parser = argparse.ArgumentParser(description='Adjust timestamps in PCAP files within a directory.')
    parser.add_argument('input_directory', help='Path to the directory containing PCAP files.')
    parser.add_argument('output_directory', help='Path where the modified PCAP files will be saved.')
    #parser.add_argument('--factor', type=int, default=2, help='Factor "n" used in timestamp adjustment.')
    parser.add_argument('--factor', type=float, default=2, help='Factor "n" used in timestamp adjustment.')



    args = parser.parse_args()

    # Ensure the output directory exists
    create_dir_if_not_exists(args.output_directory)

    # Get all pcap files in the specified directory
    pcap_files = [os.path.join(args.input_directory, f) for f in os.listdir(args.input_directory) if f.endswith('.pcap')]

    # Use multiprocessing to process pcap files in parallel with a maximum of 4 workers
    with Pool(min(8, len(pcap_files))) as pool:
        pool.map(adjust_timestamps, [(f, args.output_directory, args.factor) for f in pcap_files])

if __name__ == '__main__':
    main()



# How to use
# python3 adjust_timestamps_directory.py [input_directory] [output_directory] [--factor FACTOR]
# python3 adjust_pcap_timestamps.py "../../../UNSW-NB15/UNSW-NB15-pcap_files/pcaps_22-1-2015/" "../../../UNSW-NB15/UNSW-NB15-pcap_files_4xtimedelay/pcaps_22-1-2015/" --factor 4