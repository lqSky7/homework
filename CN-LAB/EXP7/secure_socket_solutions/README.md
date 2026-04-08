# Secure Socket Practice Solutions (EXP7)

This folder contains complete TCP client/server C solutions for all questions in:

`../secure_socket_practice_questions.md`

## Structure

- `q2_bank_activation/` (Added Question 2)
- `problem1_erp_activation/`
- `problem2_ecom_token/`
- `problem3_hospital_portal/`
- `problem4_exam_gateway/`
- `problem5_file_portal_reset/`

Each subfolder has:
- `server.c`
- `client.c`

## Build and Run

From each subfolder:

```bash
gcc server.c -o server
gcc client.c -o client
```

Run server first:

```bash
./server
```

Then run client in another terminal:

```bash
./client
```

Optional arguments:
- Server: `./server <port>`
- Client: `./client <ip> <port>`
