#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter age: " AGE

if [ $AGE -ge 18 ]; then
    STATUS="adult"
else
    STATUS="child"
fi

echo "Age: $AGE"
echo "Status: $STATUS"
