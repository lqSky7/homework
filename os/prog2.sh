#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter first number: " a
read -p "Enter second number: " b
read -p "Enter third number: " c

if [ $a -ge $b ] && [ $a -ge $c ]; then
    echo "Biggest number is: $a"
elif [ $b -ge $a ] && [ $b -ge $c ]; then
    echo "Biggest number is: $b"
else
    echo "Biggest number is: $c"
fi
