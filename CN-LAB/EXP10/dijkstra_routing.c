#include <stdio.h>
#include <limits.h>

#define MAXV 100
#define INF 1000000000

int main(void) {
    int n, graph[MAXV][MAXV], dist[MAXV], visited[MAXV] = {0}, parent[MAXV];
    int src;

    printf("Enter number of vertices: ");
    scanf("%d", &n);
    if (n <= 0 || n > MAXV) return 1;

    printf("Enter adjacency matrix (0 for no edge, positive weight otherwise):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &graph[i][j]);
            if (i != j && graph[i][j] == 0) graph[i][j] = INF;
        }
    }

    printf("Enter source vertex (0 to %d): ", n - 1);
    scanf("%d", &src);
    if (src < 0 || src >= n) return 1;

    for (int i = 0; i < n; i++) {
        dist[i] = INF;
        parent[i] = -1;
    }
    dist[src] = 0;

    for (int c = 0; c < n - 1; c++) {
        int u = -1, best = INF;
        for (int i = 0; i < n; i++) {
            if (!visited[i] && dist[i] < best) {
                best = dist[i];
                u = i;
            }
        }
        if (u == -1) break;
        visited[u] = 1;

        for (int v = 0; v < n; v++) {
            if (!visited[v] && graph[u][v] < INF && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                parent[v] = u;
            }
        }
    }

    printf("\nShortest path costs from source %d:\n", src);
    for (int i = 0; i < n; i++) {
        if (dist[i] >= INF) printf("To %d: Unreachable\n", i);
        else printf("To %d: %d\n", i, dist[i]);
    }

    return 0;
}
