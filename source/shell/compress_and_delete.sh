#!/bin/bash

# The directory to be processed
SOURCE_DIR=$1

# Check if the source directory is provided and exists
if [ -z "$SOURCE_DIR" ]; then
    echo "Usage: $0 <source-directory>"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Directory not found: $SOURCE_DIR"
    exit 1
fi

# Navigate to the source directory
cd "$SOURCE_DIR"

# Loop through each item in the directory
for ITEM in *; do
    # Check if the item is a directory and not already a .tar.gz file
    if [ -d "$ITEM" ] && [ "${ITEM: -7}" != ".tar.gz" ]; then
        # Compress the folder
        tar -czvf "$ITEM.tar.gz" "$ITEM"

        # Verify successful compression
        if [ -f "$ITEM.tar.gz" ]; then
            echo "Compression successful: $ITEM"
            
            # Delete the original folder
            rm -rf "$ITEM"
            echo "Deleted original folder: $ITEM"
        else
            echo "Compression failed: $ITEM"
        fi
    fi
done

echo "Processing complete."
