#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter array elements (space separated): " -a arr

max=${arr[0]}
sum=0

for element in "${arr[@]}"; do
    if [ $element -gt $max ]; then
        max=$element
    fi
    sum=$((sum + element))
done

echo "Greatest element: $max"
echo "Sum of elements: $sum"
