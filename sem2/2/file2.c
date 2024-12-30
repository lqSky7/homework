#include <stdio.h>

int main() {
    int a, b, c;
    float d;
    scanf("%d %d %d", &a, &b, &c);
    
    d = ((float)(c - a - b) / (a + b)) * 100;
    printf("%.2f", d);
    return 0;
}