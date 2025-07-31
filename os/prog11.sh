#!/bin/bash

read -p "Enter a string: " str

while true; do
    echo "String Operations:"
    echo "1. Length 2. Uppercase 3. Lowercase 4. Reverse"
    echo "5. First char 6. Last char 7. Substring 8. Replace 9. Exit"
    read -p "Choose operation: " choice

    case $choice in
        1) echo "Length: ${#str}" ;;
        2) echo "Uppercase: $(echo "$str" | tr '[:lower:]' '[:upper:]')" ;;
        3) echo "Lowercase: $(echo "$str" | tr '[:upper:]' '[:lower:]')" ;;
        4) echo "Reverse: $(echo $str | rev)" ;;
        5) echo "First character: ${str:0:1}" ;;
        6) echo "Last character: ${str: -1}" ;;
        7) 
            read -p "Start position (0-based): " start
            read -p "Length: " len
            echo "Substring: ${str:$start:$len}"
            ;;
        8) 
            read -p "Find: " find
            read -p "Replace with: " replace
            echo "Result: ${str//$find/$replace}"
            ;;
        9) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid choice" ;;
    esac
    echo
done
