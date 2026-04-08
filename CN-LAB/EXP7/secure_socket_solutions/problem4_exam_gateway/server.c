#include <arpa/inet.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUF 768

static int valid_student_name(const char *s) {
    int alpha_count = 0;
    if (!s || !*s) return 0;
    for (int i = 0; s[i]; i++) {
        if (isalpha((unsigned char)s[i])) alpha_count++;
        else if (s[i] != ' ') return 0;
    }
    return alpha_count >= 6;
}

static int valid_reg_no(const char *s) {
    if (strlen(s) != 10) return 0;
    for (int i = 0; i < 4; i++) if (!isdigit((unsigned char)s[i])) return 0;
    for (int i = 4; i < 7; i++) if (!(s[i] >= 'A' && s[i] <= 'Z')) return 0;
    for (int i = 7; i < 10; i++) if (!isdigit((unsigned char)s[i])) return 0;
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

static void build_access_key(const char *name, const char *reg, char *out, size_t out_sz) {
    char compact[256];
    size_t k = 0;
    for (size_t i = 0; name[i] && k + 1 < sizeof(compact); i++) {
        if (name[i] != ' ') compact[k++] = toupper((unsigned char)name[i]);
    }
    compact[k] = '\0';
    snprintf(out, out_sz, "%c%c%c%c%c%cEXM", compact[0], compact[1], compact[2], reg[7], reg[8], reg[9]);
}

int main(int argc, char *argv[]) {
    int port = (argc > 1) ? atoi(argv[1]) : 9105;
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

    printf("Exam gateway server on %d\n", port);
    while (1) {
        int cfd = accept(sfd, NULL, NULL);
        if (cfd < 0) continue;
        char in[BUF] = {0}, name[256] = {0}, reg[64] = {0}, pass[128] = {0}, out[BUF] = {0};
        int n = recv(cfd, in, sizeof(in) - 1, 0);
        if (n > 0 && sscanf(in, "%255[^|]|%63[^|]|%127[^\n]", name, reg, pass) == 3) {
            char errors[BUF] = {0};
            if (!valid_student_name(name)) strcat(errors, "student_name must be alphabets/spaces, min 6 letters excluding spaces; ");
            if (!valid_reg_no(reg)) strcat(errors, "reg_no must match YYYYXXXNNN with uppercase dept; ");
            if (strlen(pass) < 8 || !has_digit(pass) || !has_special(pass))
                strcat(errors, "password must be >=8 with digit and special char; ");

            if (errors[0]) {
                snprintf(out, sizeof(out), "ACCESS_DENIED|%.740s\n", errors);
            } else {
                char key[64];
                build_access_key(name, reg, key, sizeof(key));
                snprintf(out, sizeof(out), "ACCESS_GRANTED|access_key=%s\n", key);
            }
        } else {
            snprintf(out, sizeof(out), "ACCESS_DENIED|Invalid request format\n");
        }
        send(cfd, out, strlen(out), 0);
        close(cfd);
    }
}
