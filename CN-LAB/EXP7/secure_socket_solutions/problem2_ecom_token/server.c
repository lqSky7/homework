#include <arpa/inet.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define BUF 512

static int valid_email_username(const char *s) {
    size_t n = strlen(s);
    if (n < 5 || n > 15) return 0;
    for (size_t i = 0; i < n; i++) {
        if (!(s[i] >= 'a' && s[i] <= 'z') && !isdigit((unsigned char)s[i])) return 0;
    }
    return 1;
}

static int has_digit(const char *s) {
    for (int i = 0; s[i]; i++) if (isdigit((unsigned char)s[i])) return 1;
    return 0;
}

static int has_special(const char *s) {
    for (int i = 0; s[i]; i++) if (!isalnum((unsigned char)s[i])) return 1;
    return 0;
}

static void build_token(const char *user, char *token, size_t token_sz) {
    char reversed[160];
    size_t n = strlen(user), k = 0;
    for (size_t i = 0; i < n && k + 1 < sizeof(reversed); i++) reversed[k++] = user[n - 1 - i];
    reversed[k] = '\0';

    time_t now = time(NULL);
    struct tm tm_now;
    int day = 0;
    if (localtime_r(&now, &tm_now)) day = tm_now.tm_mday;
    snprintf(token, token_sz, "%s%d#ECOM", reversed, day);
}

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9103;
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) return 1;
    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return 1;
    if (listen(sfd, 5) < 0) return 1;

    printf("E-commerce login server on %d\n", port);
    while (1) {
        int cfd = accept(sfd, NULL, NULL);
        if (cfd < 0) continue;
        char in[BUF] = {0}, user[128] = {0}, pass[128] = {0}, out[BUF] = {0};
        int n = recv(cfd, in, sizeof(in) - 1, 0);
        if (n > 0 && sscanf(in, "%127[^|]|%127[^\n]", user, pass) == 2) {
            char reason[BUF] = {0};
            if (!valid_email_username(user))
                strcat(reason, "email_username must be lowercase letters/digits, length 5..15; ");
            if (strlen(pass) < 8 || !has_digit(pass) || !has_special(pass))
                strcat(reason, "password must be >=8 with digit and special char; ");

            if (reason[0]) {
                snprintf(out, sizeof(out), "LOGIN_REJECTED|%.480s\n", reason);
            } else {
                char token[256];
                build_token(user, token, sizeof(token));
                snprintf(out, sizeof(out), "LOGIN_ACCEPTED|token=%s\n", token);
            }
        } else {
            snprintf(out, sizeof(out), "LOGIN_REJECTED|Invalid request format\n");
        }
        send(cfd, out, strlen(out), 0);
        close(cfd);
    }
}
