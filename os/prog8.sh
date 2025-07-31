#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter a number: " num

factorial=1
original=$num

while [ $num -gt 1 ]; do
    factorial=$((factorial * num))
    num=$((num - 1))
done

echo "Factorial of $original is: $factorial"
