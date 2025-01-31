 #include<stdio.h>

int* mx(int* arr, int size){
    int *maxpt = arr;
    int* start = arr;
    int *end = arr+size-1;
    while(end>= start){
        if(*start >= *maxpt){

            maxpt = start;
        }
        start++;
    }

    return maxpt;
}

int main(){
    int si = 6;
    int arr[6] = {1,2,3,4,5,2};
    int* result = mx(arr, si);
    printf("%d", *result);
}
