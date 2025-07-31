#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter a number: " num

reverse=0
original=$num

while [ $num -gt 0 ]; do
    digit=$((num % 10))
    reverse=$((reverse * 10 + digit))
    num=$((num / 10))
done

echo "Original number: $original"
echo "Reversed number: $reverse"
