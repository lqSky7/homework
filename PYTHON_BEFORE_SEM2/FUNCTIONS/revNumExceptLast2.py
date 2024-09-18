n = str(input())
if(len(n) <=2 ):
    print("invalid")
    exit()
rev = ""
if(len(n) == 3):
    print(n)

for i in range(len(n)-2):
    rev+=n[len(n)-3-i]    
    
rev += n[-1]
    
rev += n[-2]
print(rev)

#pseudocode:

```
input a string and store it in 'n'
If the length of 'n' is less than or equal to 2:
    Print "invalid"
    Exit the program

Initialize an empty string 'rev'

If the length of 'n' is exactly 3:
    Print 'n'

For each 'i' in the range from 0 to the length of 'n' minus 2:
    Append the (length of 'n' - 3 - i)-th character of 'n' to 'rev'

Append the last character of 'n' to 'rev'
Append the second last character of 'n' to 'rev'

Print 'rev'
```
