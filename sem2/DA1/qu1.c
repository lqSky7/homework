#include <stdio.h>

void maxPos(int arr[], int n) {
    int maxSum = 0;
    int currentSum = 0;
    int start = 0, end = 0;
    int tempStart = 0;
 // double sliding window approach for start and sum....

    int output[n];
    int outputSize = 0;

    for (int i = 0; i < n; i++) {
        if (arr[i] > 0) {
            currentSum += arr[i];


            if (currentSum > maxSum) {
                maxSum = currentSum;
                start = tempStart;
                end = i;
            }
        } else {

            currentSum = 0;
            tempStart = i + 1;
        }
    }


    if (maxSum == 0) {
        printf("Max Sum : 0\nElements : {}\n");
        return;
    }

    for (int i = start; i <= end; i++) {
        output[outputSize++] = arr[i];
    }


    printf("Max Sum : %d\n", maxSum);
    printf("Elements : {");
    for (int i = 0; i < outputSize; i++) {
        printf("%d", output[i]);
        if (i < outputSize - 1) {
            printf(", ");
        }
    }
    printf("}\n");
}

int main() {
    int a;
    scanf("%d", &a);
    int arr[a];

    for(int i=0;i<a;i++){
    scanf("%d", &arr[i]);
    }

    maxPos(arr, a);

    return 0;
}
