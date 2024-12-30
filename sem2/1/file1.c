#include <stdio.h>

int main() {
    char a[50], b[50];
    int c;
    char d;
    double e;
    
    scanf("%[^\n]%*c", a);
    scanf("%[^\n]%*c", b);
    scanf("%d", &c);
    scanf(" %c", &d);
    scanf("%lf", &e);
    
    printf("%s\n%s\n%d\n%c\n%.1lf", a, b, c, d, e);
    return 0;
}