#include <stdio.h>

int main() {
    int a, b;
    scanf("%d", &a);
    scanf("%d", &b);
    
    printf("The number of students in each team is %d and the number of students left out is %d", a/b, a%b);
    return 0;
}