#!/bin/bash

# Check if a directory was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/your/images"
  exit 1
fi

# Navigate to the directory provided as an argument
cd "$1" || { echo "Directory not found: $1"; exit 1; }

# Initialize a counter
counter=0

# Loop through all JPG image files in the directory
for img in *.jpg; do
    new_name="Image_$counter.jpg"
    
    # Check if the new name already exists
    if [ -e "$new_name" ]; then
        echo "Warning: $new_name already exists, skipping..."
    else
        mv "$img" "$new_name"
        rm "$img"
        echo "Renamed $img to $new_name"
        counter=$((counter + 1))
    fi
done
for img in *.PNG; do
    new_name="Image_$counter.PNG"
    
    # Check if the new name already exists
    if [ -e "$new_name" ]; then
        echo "Warning: $new_name already exists, skipping..."
    else
        mv "$img" "$new_name"
        rm "$img"
        echo "Renamed $img to $new_name"
        counter=$((counter + 1))
    fi
done

# Uncomment this block for PNG files if needed
# counter=0  # Optional: Reset the counter for PNG files
# for img in *.png; do
#     new_name="Image_$counter.png"
#     
#     # Check if the new name already exists
#     if [ -e "$new_name" ]; then
#         echo "Warning: $new_name already exists, skipping..."
#     else
#         mv "$img" "$new_name"
#         echo "Renamed $img to $new_name"
#         counter=$((counter + 1))
#     fi
# done
