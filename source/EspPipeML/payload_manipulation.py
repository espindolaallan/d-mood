import os
import sys
import glob
from scapy.all import rdpcap, wrpcap, IP, TCP, UDP
from multiprocessing import Pool

def manipulate_headers(packet, payload_scale_factor=0.1, mtu=1500):
    if packet.haslayer(IP):
        # Predict the size change: increase or decrease
        change_amount = int(packet[IP].len * abs(payload_scale_factor))
        new_size = packet[IP].len + change_amount if payload_scale_factor > 0 else packet[IP].len - change_amount
        
        # Ensure the new size is within the minimum IP packet size (20 bytes) and MTU
        if 20 <= new_size <= mtu - 20:  # 20 bytes for typical IP header size
            packet[IP].len = new_size
            del packet[IP].chksum  # Let Scapy recalculate the checksum

            # Recalculate TCP/UDP checksums after IP len modification
            if packet.haslayer(TCP):
                del packet[TCP].chksum
            elif packet.haslayer(UDP):
                del packet[UDP].chksum
    return packet


def process_pcap(args):
    input_filepath, output_filepath, payload_scale_factor = args
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    print('Working on {}' .format(output_filepath))
    # Read packets from the input pcap file
    try:
        packets = rdpcap(input_filepath)
    except FileNotFoundError:
        print(f"Error: File {input_filepath} not found.")
        return
    
    # Manipulate headers of packets
    modified_packets = [manipulate_headers(packet, payload_scale_factor) for packet in packets]
    
    # Save the modified packets to the output pcap file
    wrpcap(output_filepath, modified_packets)
    print(f"Modified pcap saved to {output_filepath}")

def main(input_dir, output_dir, scale_factor):
    # Get list of all pcap files in the input directory
    input_files = glob.glob(os.path.join(input_dir, '*.pcap'))
    
    # Prepare arguments for the worker function
    args = [(infile, os.path.join(output_dir, os.path.basename(infile)), scale_factor) for infile in input_files]

    # Process pcap files in parallel
    with Pool(processes=8) as pool:
        pool.map(process_pcap, args)

if __name__ == "__main__":
    # Ensure the correct number of arguments
    if len(sys.argv) != 4:
        print("Usage: python3 script.py <input_dir> <output_dir> <payload_scale_percentage>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    # Convert the percentage to fraction (e.g., 10% becomes 0.1)
    payload_scale_factor = float(sys.argv[3]) / 100

    main(input_dir, output_dir, payload_scale_factor)

# How to use
# python3 payload_manipulation.py <input_directory_path> <output_directory_path> <percentage>
# python3 payload_manipulation.py '../../UNSW-NB15-delay/UNSW-NB15-pcap_files_no_payload/pcaps_17-2-2015/' '../../UNSW-NB15-delay/UNSW-NB15-pcap_files_50percent_payload/pcaps_17-2-2015/' 50