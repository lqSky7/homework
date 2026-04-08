#include <arpa/inet.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUF 768

static int only_alpha(const char *s) {
    if (!s || !*s) return 0;
    for (int i = 0; s[i]; i++) if (!isalpha((unsigned char)s[i])) return 0;
    return 1;
}

static int only_upper_alpha_len3(const char *s) {
    if (strlen(s) != 3) return 0;
    for (int i = 0; i < 3; i++) if (!(s[i] >= 'A' && s[i] <= 'Z')) return 0;
    return 1;
}

static int has_upper(const char *s) {
    for (int i = 0; s[i]; i++) if (isupper((unsigned char)s[i])) return 1;
    return 0;
}

static int has_digit(const char *s) {
    for (int i = 0; s[i]; i++) if (isdigit((unsigned char)s[i])) return 1;
    return 0;
}

static int has_special(const char *s) {
    for (int i = 0; s[i]; i++) if (!isalnum((unsigned char)s[i])) return 1;
    return 0;
}

static void reverse_append_erp(const char *in, char *out, size_t out_sz) {
    size_t n = strlen(in), k = 0;
    for (size_t i = 0; i < n && k + 1 < out_sz; i++) out[k++] = in[n - 1 - i];
    if (k + 4 < out_sz) {
        out[k++] = '@';
        out[k++] = 'E';
        out[k++] = 'R';
        out[k++] = 'P';
    }
    out[k] = '\0';
}

static void mask_password(const char *in, char *out, size_t out_sz) {
    size_t n = strlen(in);
    if (n == 0) {
        out[0] = '\0';
        return;
    }
    if (n == 1) {
        snprintf(out, out_sz, "%c", in[0]);
        return;
    }
    size_t k = 0;
    out[k++] = in[0];
    for (size_t i = 1; i + 1 < n && k + 1 < out_sz; i++) out[k++] = '*';
    if (k + 1 < out_sz) out[k++] = in[n - 1];
    out[k] = '\0';
}

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9102;
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) return 1;
    int opt = 1;
    if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        return 1;
    }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) return 1;
    if (listen(sfd, 5) < 0) return 1;

    printf("ERP activation server on %d\n", port);
    while (1) {
        int cfd = accept(sfd, NULL, NULL);
        if (cfd < 0) continue;

        char in[BUF] = {0}, username[128] = {0}, password[128] = {0}, dept[32] = {0}, out[BUF] = {0};
        int n = recv(cfd, in, sizeof(in) - 1, 0);
        if (n > 0 && sscanf(in, "%127[^|]|%127[^|]|%31[^\n]", username, password, dept) == 3) {
            char errors[BUF] = {0};
            if (strlen(username) < 6 || !only_alpha(username))
                strcat(errors, "Username: minimum 6 alphabets only; ");
            if (strlen(password) < 10 || !has_upper(password) || !has_digit(password) || !has_special(password))
                strcat(errors, "Password: min 10 with uppercase, digit, special; ");
            if (!only_upper_alpha_len3(dept))
                strcat(errors, "Department code: exactly 3 uppercase letters; ");

            if (errors[0]) {
                snprintf(out, sizeof(out), "ACCOUNT_REJECTED|%.730s\n", errors);
            } else {
                char enc_user[160], masked[160];
                reverse_append_erp(username, enc_user, sizeof(enc_user));
                mask_password(password, masked, sizeof(masked));
                snprintf(out, sizeof(out), "ACCOUNT_CREATED|encrypted_username=%s|masked_password=%s|department_code=%s\n", enc_user, masked, dept);
            }
        } else {
            snprintf(out, sizeof(out), "ACCOUNT_REJECTED|Invalid request format\n");
        }
        send(cfd, out, strlen(out), 0);
        close(cfd);
    }
}
