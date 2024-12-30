#include <stdio.h>

int main() {
    char a;
    scanf(" %c", &a);
    
    switch(a) {
        case 'O':
            printf("above 90%%");
            break;
        case 'A':
            printf("above 80%%");
            break;
        case 'B':
            printf("above 70%%");
            break;
        case 'C':
            printf("above 60%%");
            break;
        case 'D':
            printf("above 50%%");
            break;
        case 'U':
            printf("fail");
            break;
    }
    return 0;
}