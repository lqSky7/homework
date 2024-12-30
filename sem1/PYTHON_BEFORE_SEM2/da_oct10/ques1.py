physics = [1, 0, 1, 0, 1]
math = [0, 1, 1, 0, 0]
chemistry = [0, 0, 1, 0, 1]
computer_science = [1, 0, 0, 1, 1]

total_failures = 0
for i in range(len(physics)):
    if physics[i] == 1 or math[i] == 1 or chemistry[i] == 1 or computer_science[i] == 1:
        total_failures += 1

print(" number  of students who failed total_failures)

