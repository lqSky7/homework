#include <stdio.h>

int main() {
    char a[50];
    int b;
    char c;
    
    scanf("%s", a);
    scanf("%d", &b);
    scanf(" %c", &c);
    
    if (b < 12 || c == 'f') {
        printf("concession");
    } else {
        printf("no concession");
    }
    return 0;
}