with open('example.txt', 'w') as file:
    file.write('Hello, world!\n')
    file.write('This is a test file.\n')
    file.write('Each line will be read and processed.\n')

with open('example.txt', 'r') as file:
    print(file.read())

with open('example.txt', 'r') as file:
    print(file.readline())

with open('example.txt', 'r') as file:
    print(file.readlines())

with open('example.txt', 'r') as file:
    print(file.tell())
    file.seek(7)
    print(file.tell())
    print(file.read())