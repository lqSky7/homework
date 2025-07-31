#!/bin/bash
echo "24BYB1095 Diljot Singh"

is_armstrong() {
    local num=$1
    local original=$num
    local sum=0
    local digits=0
    
    temp=$num
    while [ $temp -gt 0 ]; do
        digits=$((digits + 1))
        temp=$((temp / 10))
    done
    
    temp=$num
    while [ $temp -gt 0 ]; do
        digit=$((temp % 10))
        power=1
        for ((i=1; i<=digits; i++)); do
            power=$((power * digit))
        done
        sum=$((sum + power))
        temp=$((temp / 10))
    done
    
    if [ $sum -eq $original ]; then
        return 1
    else
        return 0
    fi
}

read -p "Enter a number: " number
is_armstrong $number

if [ $? -eq 1 ]; then
    echo "$number is an Armstrong number"
else
    echo "$number is not an Armstrong number"
fi
