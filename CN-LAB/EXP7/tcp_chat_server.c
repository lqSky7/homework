#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/select.h>

#define BUF 512

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9002;
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) return 1;

    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return 1;
    if (listen(sfd, 1) < 0) return 1;

    printf("Chat server listening on %d\n", port);
    int cfd = accept(sfd, NULL, NULL);
    if (cfd < 0) return 1;
    printf("Client connected. Type /quit to end.\n");

    fd_set readfds;
    char buf[BUF];
    while (1) {
        FD_ZERO(&readfds);
        FD_SET(0, &readfds);
        FD_SET(cfd, &readfds);
        int maxfd = cfd;

        if (select(maxfd + 1, &readfds, NULL, NULL, NULL) < 0) break;

        if (FD_ISSET(0, &readfds)) {
            if (!fgets(buf, sizeof(buf), stdin)) break;
            send(cfd, buf, strlen(buf), 0);
            if (strncmp(buf, "/quit", 5) == 0) break;
        }

        if (FD_ISSET(cfd, &readfds)) {
            int n = recv(cfd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) break;
            buf[n] = '\0';
            printf("Client: %s", buf);
            if (strncmp(buf, "/quit", 5) == 0) break;
        }
    }

    close(cfd);
    close(sfd);
    return 0;
}
