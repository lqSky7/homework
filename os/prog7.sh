#!/bin/bash
echo "24BYB1095 Diljot Singh"

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

sum=0
count=0

for arg in "$@"; do
    sum=$((sum + arg))
    count=$((count + 1))
done

average=$((sum / count))
echo "Sum: $sum"
echo "Count: $count"
echo "Average: $average"
