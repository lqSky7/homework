#!/bin/bash
echo "24BYB1095 Diljot Singh"
read -p "Enter first number: " a
read -p "Enter second number: " b
read -p "Enter operation (+, -, *, /): " op

case $op in
    +) result=$((a + b))
       echo "Result: $result" ;;
    -) result=$((a - b))
       echo "Result: $result" ;;
    *) result=$((a * b))
       echo "Result: $result" ;;
    /) if [ $b -ne 0 ]; then
           result=$((a / b))
           echo "Result: $result"
       else
           echo "Division by zero error"
       fi ;;
    *) echo "Invalid operation" ;;
esac
