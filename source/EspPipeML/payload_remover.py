import os
import sys
import glob
from scapy.all import rdpcap, wrpcap, Raw
from multiprocessing import Pool

def remove_payload(packet):
    if packet.haslayer(Raw):
        packet[Raw].load = b''  # Replace the payload with empty bytes
    return packet

def process_pcap(args):
    input_filepath, output_filepath = args
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    print('Working on {}' .format(output_filepath))
    # Read packets from the input pcap file
    try:
        packets = rdpcap(input_filepath)
    except FileNotFoundError:
        print(f"Error: File {input_filepath} not found.")
        return
    
    # Remove payloads from packets
    modified_packets = [remove_payload(packet) for packet in packets]
    
    # Save the modified packets to the output pcap file
    wrpcap(output_filepath, modified_packets)
    print(f"Modified pcap saved to {output_filepath}")

def main(input_dir, output_dir):
    # Get list of all pcap files in input directory
    input_files = glob.glob(os.path.join(input_dir, '*.pcap'))
    
    # Prepare arguments for worker function
    args = [(infile, os.path.join(output_dir, os.path.basename(infile))) for infile in input_files]

    # Process pcap files in parallel
    with Pool(processes=8) as pool:
        pool.map(process_pcap, args)

if __name__ == "__main__":
    # Ensure correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python3 script.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    main(input_dir, output_dir)

# How to use
# python3 payload_remover.py '../../UNSW-NB15-pcap_files/pcaps_22-1-2015/' '../../UNSW-NB15-pcap_files_no_payload/pcaps_22-1-2015'