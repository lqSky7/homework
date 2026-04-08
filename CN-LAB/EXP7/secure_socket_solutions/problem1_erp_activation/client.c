#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUF 768

static void trim_newline(char *s) {
    size_t n = strlen(s);
    if (n && s[n - 1] == '\n') s[n - 1] = '\0';
}

int main(int argc, char *argv[]) {
    const char *ip = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? atoi(argv[2]) : 9102;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 1;

    struct sockaddr_in server = {0};
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    inet_pton(AF_INET, ip, &server.sin_addr);

    if (connect(fd, (struct sockaddr *)&server, sizeof(server)) < 0) return 1;

    char username[128], password[128], dept[32], req[BUF], resp[BUF];
    printf("Username: ");
    if (!fgets(username, sizeof(username), stdin)) return 1;
    printf("Password: ");
    if (!fgets(password, sizeof(password), stdin)) return 1;
    printf("Department code (3 uppercase letters): ");
    if (!fgets(dept, sizeof(dept), stdin)) return 1;

    trim_newline(username);
    trim_newline(password);
    trim_newline(dept);

    snprintf(req, sizeof(req), "%s|%s|%s\n", username, password, dept);
    send(fd, req, strlen(req), 0);

    int n = recv(fd, resp, sizeof(resp) - 1, 0);
    if (n > 0) {
        resp[n] = '\0';
        printf("%s", resp);
    }
    close(fd);
    return 0;
}

