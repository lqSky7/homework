# CN-LAB Assignment Pack

This folder contains detailed material for all 10 CN lab experiments.

## Folder map
- `EXP1` Networking basics notes + command table template
- `EXP2` Error detection mechanisms in C (VRC, LRC, CRC, Checksum)
- `EXP3` Hamming code (sender/receiver)
- `EXP4` Lab activity report template (topology, DHCP, shared storage)
- `EXP5` Lab activity report template (website + SNMP tool manual)
- `EXP6` IPv4 conversion and subnetting (classful/classless)
- `EXP7` TCP client/server programs (math + chat)
  - Includes additional secure socket practice questions
- `EXP8` ARQ flow control programs with socket-based simulation
- `EXP9` UDP multiuser chat program
- `EXP10` Routing algorithms (Dijkstra and Bellman-Ford)

## Compile examples
```bash
gcc CN-LAB/EXP2/error_detection.c -o CN-LAB/EXP2/error_detection
gcc CN-LAB/EXP3/hamming_code.c -o CN-LAB/EXP3/hamming_code
gcc CN-LAB/EXP6/ip_addressing.c -lm -o CN-LAB/EXP6/ip_addressing
gcc CN-LAB/EXP10/dijkstra_routing.c -o CN-LAB/EXP10/dijkstra_routing
gcc CN-LAB/EXP10/bellman_ford_routing.c -o CN-LAB/EXP10/bellman_ford_routing
```

Socket-based files may need two terminals (server and client).

## Extra practice question bank
- `EXP7/secure_socket_practice_questions.md`:
  secure credential-handling and validation-based client-server questions.

## Important submission note
For experiments requiring lab photos/screenshots, use your own photos and replace placeholders in template files.
