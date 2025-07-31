#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter 5 numbers (space separated): " -a arr

for ((i = 0; i < 4; i++)); do
    for ((j = 0; j < 4 - i; j++)); do
        if [[ ${arr[j]} -lt ${arr[$((j+1))]} ]]; then
            temp=${arr[j]}
            arr[j]=${arr[$((j+1))]}
            arr[$((j+1))]=$temp
        fi
    done
done

echo "Numbers in descending order: ${arr[@]}"

