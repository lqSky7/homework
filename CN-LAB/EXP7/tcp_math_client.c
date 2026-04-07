#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define BUF 256

int main(int argc, char *argv[]) {
    const char *ip = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : 9001;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 1;

    struct sockaddr_in server = {0};
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, ip, &server.sin_addr);

    if (connect(fd, (struct sockaddr *)&server, sizeof(server)) < 0) return 1;

    char line[BUF], resp[BUF];
    printf("Connected. Enter expression like: 10 + 20 (type quit to exit)\n");
    while (1) {
        printf("> ");
        if (!fgets(line, sizeof(line), stdin)) break;
        if (strncmp(line, "quit", 4) == 0) break;

        send(fd, line, strlen(line), 0);
        int n = recv(fd, resp, sizeof(resp) - 1, 0);
        if (n <= 0) break;
        resp[n] = '\0';
        printf("%s", resp);
    }

    close(fd);
    return 0;
}
