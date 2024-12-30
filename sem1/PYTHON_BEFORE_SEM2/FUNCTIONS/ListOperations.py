# we have init list, ask for number to perform action, delete, append, view
li = ["cookie", "cake", "chocolate"]
ll =1

while(ll == 1):
    # main
    i = int(input("Enter 1 to view list, 2 to append list, 3 to delete some element"))
    if(i == 1):
        print(li)
        
    if(i == 2):
        k = str(input("Enter the element you want to add"))
        li.append(k)

    if(i == 3):
        k = str(input("Enter the element you want to add"))
        li.remove(k)
    
    ll = int(input("press 1 to run the loop again, 0 to exit"))
    if(ll==0):
        break


#PSEUDOCODE 
# Initialize list 'li' with ["cookie", "cake", "chocolate"]
# Set 'll' to 1

# While 'll' equals 1:
#     Ask user for input (1 to view list, 2 to append, 3 to delete)
    
#     If input is 1:
#         Display the list 'li'
    
#     If input is 2:
#         Ask user for an element to add
#         Append the element to the list 'li'
    
#     If input is 3:
#         Ask user for an element to delete
#         Remove the element from the list 'li'
    
#     Ask user if they want to continue (1 to repeat, 0 to exit)
    
#     If user input is 0:
#         Exit the loop
