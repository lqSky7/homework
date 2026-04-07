#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/select.h>

#define BUF 512

int main(int argc, char *argv[]) {
    const char *ip = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : 9003;
    char name[32];

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return 1;

    struct sockaddr_in server = {0};
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, ip, &server.sin_addr);

    printf("Enter username: ");
    scanf("%31s", name);
    getchar();

    char joinmsg[BUF];
    snprintf(joinmsg, sizeof(joinmsg), "/join %s", name);
    sendto(fd, joinmsg, strlen(joinmsg), 0, (struct sockaddr *)&server, sizeof(server));

    printf("Connected to UDP chat. Type /quit to exit.\n");
    while (1) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(0, &fds);
        FD_SET(fd, &fds);

        if (select(fd + 1, &fds, NULL, NULL, NULL) < 0) break;

        if (FD_ISSET(0, &fds)) {
            char line[BUF];
            if (!fgets(line, sizeof(line), stdin)) break;
            line[strcspn(line, "\n")] = 0;
            if (strcmp(line, "/quit") == 0) break;
            sendto(fd, line, strlen(line), 0, (struct sockaddr *)&server, sizeof(server));
        }

        if (FD_ISSET(fd, &fds)) {
            char buf[BUF];
            int n = recvfrom(fd, buf, sizeof(buf) - 1, 0, NULL, NULL);
            if (n > 0) {
                buf[n] = '\0';
                printf("%s\n", buf);
            }
        }
    }

    close(fd);
    return 0;
}
