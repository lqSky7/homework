# EXP7 Extra Practice: Secure Client-Server Problems

## Added question (from provided image)

### Question 2 [15 Marks]
Consider a secure bank server that manages user credentials for net banking accounts using a client-server socket programming model. When a user activates their account, the client sends the username and password to the server. The server performs the following operations.

**Validation Rules**
- Username must be at least 6 characters long and contain only alphabets.
- Password must be at least 8 characters long and include at least one digit and one special character.

**Username Encryption**
- Reverse the username.
- Append the string `"123"` at the end.

> Implement the sender (client) and receiver (server) logic in C using sockets, apply validation at the server side, store/transmit only processed values, and display clear success/error responses.

---

## Similar type of problems

### Problem 1: University ERP Account Activation (15 Marks)
A university ERP server receives `username`, `password`, and `department_code` from a TCP client.

**Rules**
- Username: minimum 6 alphabets only.
- Password: minimum 10 chars, at least one uppercase, one digit, one special character.
- Department code: exactly 3 uppercase letters.

**Processing**
- Username encryption: reverse username and append `"@ERP"`.
- Password masking before storage: keep first and last character visible, mask others with `*`.

**Task**
Build client-server C programs. Server validates, processes, and responds with either:
- `ACCOUNT_CREATED` and processed output, or
- error list with failed validation rules.

---

### Problem 2: E-Commerce Login Token Generator (15 Marks)
Client sends `email_username` and `password` to server.

**Rules**
- email_username: only lowercase letters and digits, length 5 to 15.
- password: minimum 8 chars, at least one digit and one special character.

**Processing**
- Create login token by:
  1. reversing `email_username`,
  2. appending current day number (from server date),
  3. appending `"#ECOM"`.

**Task**
Implement TCP socket client/server in C where server returns:
- `LOGIN_ACCEPTED` with token, or
- `LOGIN_REJECTED` with reason.

---

### Problem 3: Hospital Portal Registration (15 Marks)
Client sends `patient_id`, `username`, and `password`.

**Rules**
- patient_id: exactly 6 digits.
- username: alphabets only, minimum 6 chars.
- password: minimum 8 chars, at least one digit, one special character.

**Processing**
- Encrypted username = uppercase(username reversed) + `"HSP"`.
- Check duplicate patient_id from in-memory list at server.

**Task**
Write C socket programs (client/server) to handle validation, transformation, duplicate check, and status messages.

---

### Problem 4: Online Exam Access Gateway (15 Marks)
Client sends `student_name`, `reg_no`, `password`.

**Rules**
- student_name: alphabets and spaces only, minimum 6 chars (excluding spaces).
- reg_no: format `YYYYXXXNNN` (year + dept + number).
- password: minimum 8 chars with at least one digit and one special character.

**Processing**
- Generate access key as:
  - first 3 letters of student name (without spaces, uppercase),
  - last 3 digits of reg_no,
  - suffix `"EXM"`.

**Task**
Implement server-side validation and return either generated access key or detailed input errors.

---

### Problem 5: Secure File Portal First Login (15 Marks)
Client sends `username`, `temp_password`, `new_password`.

**Rules**
- username: minimum 6 alphabets.
- temp_password must match stored temporary value for that user.
- new_password: minimum 10 chars, must contain one uppercase, one digit, one special character.

**Processing**
- On success, server stores transformed username (reversed + `"123"`) and marks password reset complete.
- Server sends an acknowledgement with timestamp.

**Task**
Write client/server socket program in C for the first-login reset workflow with proper success/failure handling.

---

## Optional evaluation checklist for all above problems
- Input validation completeness
- Correct transformation/encryption logic
- Reliable message exchange between client and server
- Clear and user-friendly error messages
- Clean modular C functions (validation, processing, communication)
