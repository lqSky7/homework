#!/bin/bash
echo "24BYB1095 Diljot Singh"

is_palindrome() {
    local str="$1"
    local reversed=$(echo "$str" | rev)
    if [ "$str" = "$reversed" ]; then
        return 1
    else
        return 0
    fi
}

read -p "Enter strings (space separated): " -a strings
palindrome_count=0

for string in "${strings[@]}"; do
    is_palindrome "$string"
    if [ $? -eq 1 ]; then
        echo "$string is a palindrome"
        palindrome_count=$((palindrome_count + 1))
    else
        echo "$string is not a palindrome"
    fi
done

echo "Total palindromes found: $palindrome_count"
