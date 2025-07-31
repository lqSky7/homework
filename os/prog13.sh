#!/bin/bash
echo "24BYB1095 Diljot Singh"

compute_sum() {
    local sum=0
    for value in "$@"; do
        sum=$((sum + value))
    done
    echo $sum
}

read -p "Enter numbers (space separated): " -a numbers
result=$(compute_sum "${numbers[@]}")
echo "Sum of the numbers: $result"
