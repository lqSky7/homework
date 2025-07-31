#!/bin/bash
echo "24BYB1095 Diljot Singh"
count=0

echo "Enter numbers (enter 0 to stop):"
while true; do
    read -p "Enter number: " num
    if [ $num -eq 0 ]; then
        break
    fi
    if [ $num -eq 100 ]; then
        count=$((count + 1))
    fi
done

echo "Number 100 was entered $count times"
