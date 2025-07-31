#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter source directory: " source_dir
read -p "Enter backup directory: " backup_dir

if [ ! -d "$source_dir" ]; then
    echo "Source directory does not exist"
    exit 1
fi

if [ ! -d "$backup_dir" ]; then
    mkdir -p "$backup_dir"
    echo "Created backup directory: $backup_dir"
fi

timestamp=$(date +"%Y%m%d_%H%M%S")
backup_name="backup_$timestamp.tar.gz"

tar -czf "$backup_dir/$backup_name" -C "$source_dir" .

if [ $? -eq 0 ]; then
    echo "Backup created successfully: $backup_dir/$backup_name"
    echo "Backup size: $(du -h "$backup_dir/$backup_name" | cut -f1)"
else
    echo "Backup failed"
fi
