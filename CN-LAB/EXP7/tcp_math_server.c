#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define BUF 256

static double eval(double a, char op, double b, int *ok) {
    *ok = 1;
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': if (b == 0) { *ok = 0; return 0; } return a / b;
        default: *ok = 0; return 0;
    }
}

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9001;
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return 1;

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return 1;
    if (listen(server_fd, 5) < 0) return 1;

    printf("Math server listening on port %d\n", port);
    while (1) {
        int cfd = accept(server_fd, NULL, NULL);
        if (cfd < 0) continue;

        char buf[BUF];
        while (1) {
            memset(buf, 0, sizeof(buf));
            int n = recv(cfd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) break;

            double a, b;
            char op;
            int ok;
            char out[BUF];
            if (sscanf(buf, "%lf %c %lf", &a, &op, &b) == 3) {
                double ans = eval(a, op, b, &ok);
                if (ok) snprintf(out, sizeof(out), "Result: %.4lf\n", ans);
                else snprintf(out, sizeof(out), "Error: invalid operation\n");
            } else {
                snprintf(out, sizeof(out), "Error: use format <num1 op num2>\n");
            }
            send(cfd, out, strlen(out), 0);
        }
        close(cfd);
    }
}
