#!/bin/bash

# Configuration variables
executable="./nse-cbl-decay-les"
log_file="execution_log.log"
base_filename="config-plume-ex_"
init_filename="config"
file_extension=".txt"  # Change this to match your file type
num_iterations=99      # Change this to desired number of iterations

# Check if executable exists and is executable
if [ ! -x "$executable" ]; then
    echo "Error: $executable not found or not executable"
    exit 1
fi

# Check if original file exists
# if [ ! -f "${base_filename}${file_extension}" ]; then
#     echo "Error: ${base_filename}${file_extension} not found"
#     exit 1
# fi

# Initialize log file with header
echo "=== Execution Log Started at $(date) ===" > "$log_file"

# Function to clean up temporary files on script exit
# cleanup() {
#     echo "Cleaning up temporary files..."
#     # Remove any temporary numbered files
#     rm -f "${base_filename}"_*"${file_extension}"
#     echo "=== Execution Log Ended at $(date) ===" >> "$log_file"
# }

# # Register cleanup function to run on script exit
# trap cleanup EXIT

# Main execution loop
i=0
while [ $i -le $num_iterations ]
do
    echo "Processing iteration $i..."
    
    # Create new filename with number
    new_filename="${init_filename}${file_extension}"
    
    # Copy original file to new numbered filename
    cp "${base_filename}${i}${file_extension}" "$new_filename"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create $new_filename"
        continue
    fi
    
    echo "=== Execution #$i ===" >> "$log_file"
    echo "Timestamp: $(date)" >> "$log_file"
    echo "File: $new_filename" >> "$log_file"
    
    # Run the executable with the numbered file
    mpirun -np 4 $executable -arch gpu >> "$log_file" 2>&1

    
    # Log the exit status
    exit_status=$?
    echo "Exit status: $exit_status" >> "$log_file"
    echo "" >> "$log_file"  # Empty line for readability
    
    # Optional: Remove the numbered file immediately if you don't need it
    # rm "$new_filename"
    
    # Optional: Add a delay between iterations
    i=$(( $i + 1 ))
done

echo "Completed $num_iterations iterations. Results written to $log_file"