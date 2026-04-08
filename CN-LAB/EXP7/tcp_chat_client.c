#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/select.h>

#define BUF 512

int main(int argc, char *argv[]) {
    const char *ip = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : 9002;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 1;

    struct sockaddr_in server = {0};
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, ip, &server.sin_addr);

    if (connect(fd, (struct sockaddr *)&server, sizeof(server)) < 0) return 1;
    printf("Connected to chat server. Type /quit to exit.\n");

    fd_set readfds;
    char buf[BUF];
    while (1) {
        FD_ZERO(&readfds);
        FD_SET(0, &readfds);
        FD_SET(fd, &readfds);

        if (select(fd + 1, &readfds, NULL, NULL, NULL) < 0) break;

        if (FD_ISSET(0, &readfds)) {
            if (!fgets(buf, sizeof(buf), stdin)) break;
            send(fd, buf, strlen(buf), 0);
            if (strncmp(buf, "/quit", 5) == 0) break;
        }

        if (FD_ISSET(fd, &readfds)) {
            int n = recv(fd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) break;
            buf[n] = '\0';
            printf("Server: %s", buf);
            if (strncmp(buf, "/quit", 5) == 0) break;
        }
    }

    close(fd);
    return 0;
}
