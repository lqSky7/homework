#include <stdio.h>
#include <limits.h>

void firstFit(int blocks[], int m, int process[], int n) {
    int allocation[n];
    for(int i=0; i<n; i++) allocation[i] = -1;
    
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            if(blocks[j] >= process[i]) {
                allocation[i] = j;
                blocks[j] -= process[i];
                break;
            }
        }
    }
    
    printf("\nFirst Fit:\nProcess\tSize\tBlock\tFragment\n");
    for(int i=0; i<n; i++)
        printf("%d\t%d\t%d\t%d\n", i+1, process[i], 
               allocation[i]+1, allocation[i]!=-1 ? blocks[allocation[i]] : 0);
}

void bestFit(int blocks[], int m, int process[], int n) {
    int allocation[n];
    for(int i=0; i<n; i++) allocation[i] = -1;
    
    for(int i=0; i<n; i++) {
        int idx = -1, min = INT_MAX;
        for(int j=0; j<m; j++) {
            if(blocks[j] >= process[i] && blocks[j] < min) {
                min = blocks[j];
                idx = j;
            }
        }
        if(idx != -1) {
            allocation[i] = idx;
            blocks[idx] -= process[i];
        }
    }
    
    printf("\nBest Fit:\nProcess\tSize\tBlock\tFragment\n");
    for(int i=0; i<n; i++)
        printf("%d\t%d\t%d\t%d\n", i+1, process[i], 
               allocation[i]+1, allocation[i]!=-1 ? blocks[allocation[i]] : 0);
}

void worstFit(int blocks[], int m, int process[], int n) {
    int allocation[n];
    for(int i=0; i<n; i++) allocation[i] = -1;
    
    for(int i=0; i<n; i++) {
        int idx = -1, max = -1;
        for(int j=0; j<m; j++) {
            if(blocks[j] >= process[i] && blocks[j] > max) {
                max = blocks[j];
                idx = j;
            }
        }
        if(idx != -1) {
            allocation[i] = idx;
            blocks[idx] -= process[i];
        }
    }
    
    printf("\nWorst Fit:\nProcess\tSize\tBlock\tFragment\n");
    for(int i=0; i<n; i++)
        printf("%d\t%d\t%d\t%d\n", i+1, process[i], 
               allocation[i]+1, allocation[i]!=-1 ? blocks[allocation[i]] : 0);
}

int main() {
    int blocks[] = {300, 600, 350, 200, 750, 125};
    int process[] = {115, 500, 358, 200, 375};
    int m = 6, n = 5;
    
    int b1[6], b2[6], b3[6];
    for(int i=0; i<m; i++) b1[i] = b2[i] = b3[i] = blocks[i];
    
    firstFit(b1, m, process, n);
    bestFit(b2, m, process, n);
    worstFit(b3, m, process, n);
    
    return 0;
}

