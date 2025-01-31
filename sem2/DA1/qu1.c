#include <stdio.h>

int main() {
    int arr[] =  {-1,11,15, -10,30};
    int n = sizeof(arr)/sizeof(arr[0]);
    int max_sum = 0, current_sum = 0;
    int max_start = -1, max_end = -1;
    int current_start = -1, current_end = -1;
    int temp[n];
    
    for(int i=0; i<n; i++) {
        if(arr[i] > 0) {
            current_sum += arr[i];
            if(current_start == -1) current_start = i;
            current_end = i;
            if(current_sum > max_sum) {
                max_sum = current_sum;
                max_start = current_start;
                max_end = current_end;
            }
        } else {
            current_sum = 0;
            current_start = -1;
        }
    }
    
    int k=0;
    if(max_start != -1) {
        for(int i=max_start; i<=max_end; i++) {
            temp[k++] = arr[i];
        }
    }
    
    printf("Max Sum : %d\nElements : {", max_sum);
    for(int i=0; i<k; i++) {
        printf("%d", temp[i]);
        if(i != k-1) printf(",");
    }
    printf("}\n%d", k);
    return 0;
}
