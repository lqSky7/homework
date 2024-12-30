def get_student_details(n):
    students = []
    for i in range(n):
        name = input()
        place = input()
        score = float(input())
        students.append((name, place, score))
    return students

def bubble_sort_students(students):
    n = len(students)
    for i in range(n):
        for j in range(0, n-i-1):
            if students[j][2] < students[j+1][2]:
                students[j], students[j+1] = students[j+1], students[j]
    return students

n = int(input())
students = get_student_details(n)
sorted_students = bubble_sort_students(students)
print(sorted_students)