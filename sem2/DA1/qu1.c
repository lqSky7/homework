#include <stdio.h>

int max(int a, int b){
    if(a>=b){return a;}
    return b;
}
void doi(int arr[], int n) {
// sliding window approach
  int current_sum = 0;
  int max_sum = 0;


  int start = 0; int end = 0; int counter = 0;

  for(int i=0;i<n;i++){
      if(arr[i]>=0){
          counter++;
          current_sum += arr[i];
      }
      // if (arr[i]<0) {
      //     printf("hii");
      // }
      else{
          end = i-1;
          start = i-counter;
          counter = 0;
          max_sum = max(max_sum, current_sum);
          current_sum = 0;
      }
  }
  max_sum = max(max_sum, current_sum);
  printf("Max sum is: %d\n", max_sum);
  // printf("start idx is: %d\n", start);printf("end idx is: %d\n", end);

  // print elements that are counted.
  printf("Elements: {");
  for(int i = start; i<=end;i++){
      printf("%d", arr[i]);
      if (i <= end - 1) {
                  printf(", ");
              }
  }
  printf("}\n");
}

int main() {
    int n;
    printf("Enter size of array: ");
    scanf("%d", &n);


    int arr[n];
    printf("Enter elements of array: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    doi(arr, n);
    return 0;
}
