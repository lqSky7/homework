#include <stdio.h>

int main() {
    int a, b, c, d;
    scanf("%d", &a);
    scanf("%d", &b);
    scanf("%d", &c);
    scanf("%d", &d);
    
    printf("%.1f\n", (a + c) / 2.0);
    printf("%.1f", (b + d) / 2.0);
    return 0;
}