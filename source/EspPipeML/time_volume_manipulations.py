import os
import sys
import glob
from scapy.all import rdpcap, wrpcap, IP, TCP, UDP
from multiprocessing import Pool

def manipulate_packets(packets, factor):
# Ensure there are packets to manipulate
    if len(packets) == 0:
        return packets

    # Adjust the first packet's time as a reference
    time_stamp_first_pkt = packets[0].time

    # Manipulate each packet
    for i, packet in enumerate(packets):
        if i == 0:  # Skip the first packet
            continue
        difference = packet.time - time_stamp_first_pkt
        # Update the current packet's timestamp
        updated_time_stamp = (difference * (1 + factor / 100)) + time_stamp_first_pkt
        packet.time = max(updated_time_stamp, time_stamp_first_pkt)  # Prevent negative or non-sequential timestamp

        # Volume Manipulationz
        if packet.haslayer(IP):
            change_amount = int(packet[IP].len * factor / 100)
            new_size = max(min(packet[IP].len + change_amount, 1500 - 20), 20)  # Adjust between 20 bytes and MTU-20
            packet[IP].len = new_size

            # Let Scapy recalculate the checksum
            del packet[IP].chksum

            # Recalculate TCP/UDP checksums after IP len modification
            if packet.haslayer(TCP):
                del packet[TCP].chksum
            elif packet.haslayer(UDP):
                del packet[UDP].chksum

    return packets

def process_pcap(args):
    input_filepath, output_filepath, factor = args
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    print('Processing {}'.format(input_filepath))
    # Read packets from the input pcap file
    try:
        packets = rdpcap(input_filepath)
    except FileNotFoundError:
        print(f"Error: File {input_filepath} not found.")
        return
    
    # Manipulate headers of packets
    modified_packets = manipulate_packets(packets, factor)

    # Save the modified packets to the output pcap file
    wrpcap(output_filepath, modified_packets)
    print(f"Modified pcap saved to {output_filepath}")

def main(input_dir, output_dir, factor):
    # Get list of all pcap files in the input directory
    input_files = glob.glob(os.path.join(input_dir, '*.pcap'))

    # Prepare arguments for the worker function
    args = [(infile, os.path.join(output_dir, os.path.basename(infile)), factor) for infile in input_files]

    with Pool(processes=8) as pool:
        pool.map(process_pcap, args)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 script.py <input_dir> <output_dir> <factor>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    factor = float(sys.argv[3])

    main(input_dir, output_dir, factor)

# Usage
# python3 time_volume_manipulations.py <input_directory_path> <output_directory_path> <factor>
# Example: python3 time_volume_manipulations.py 'input_dir' 'output_dir' 10
