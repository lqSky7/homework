#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter filename: " filename

if [ -f "$filename" ]; then
    echo "File $filename exists"
    echo "Word frequency:"
    tr -s ' ' '\n' < "$filename" | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -nr
else
    echo "File $filename does not exist"
fi
