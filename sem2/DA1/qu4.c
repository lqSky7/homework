#include <stdio.h>
int main(){
    int n; scanf("%d",&n);
    for(int i=0;i<n;i++){
        for(int m=i;m>0;m--){
            printf("%d ",n-m+1);
        }
        for(int j=0;j<n-i;j++){
            printf("%d ",j+1);
        }
        printf("\n");
    }
}
