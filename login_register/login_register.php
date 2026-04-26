<?php
session_start();
require_once 'config.php';

/* =========================
   REGISTER
========================= */
if (isset($_POST['register'])) {

    $name = trim($_POST['name']);
    $email = trim($_POST['email']);
    $password = $_POST['password'];

    // Basic validation
    if (empty($name) || empty($email) || empty($password)) {
        $_SESSION['register_error'] = 'All fields are required!';
        $_SESSION['active_form'] = 'register';
        header("Location: index.php");
        exit();
    }

    // Check if email exists (prepared statement)
    $stmt = $conn->prepare("SELECT id FROM users WHERE email = ?");
    $stmt->bind_param("s", $email);
    $stmt->execute();
    $stmt->store_result();

    if ($stmt->num_rows > 0) {
        $_SESSION['register_error'] = 'Email is already registered!';
        $_SESSION['active_form'] = 'register';
        $stmt->close();
        header("Location: index.php");
        exit();
    }
    $stmt->close();

    // Hash password
    $hashedPassword = password_hash($password, PASSWORD_DEFAULT);

    // Force role = user
    $role = 'user';

    // Insert user
    $stmt = $conn->prepare("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)");
    $stmt->bind_param("ssss", $name, $email, $hashedPassword, $role);

    if ($stmt->execute()) {
        // Redirect after successful registration
        header("Location: index.php");
        exit();
    } else {
        $_SESSION['register_error'] = 'Something went wrong. Try again!';
        $_SESSION['active_form'] = 'register';
        header("Location: index.php");
        exit();
    }
}


/* =========================
   LOGIN
========================= */
if (isset($_POST['login'])) {

    $email = trim($_POST['email']);
    $password = $_POST['password'];

    // Basic validation
    if (empty($email) || empty($password)) {
        $_SESSION['login_error'] = 'All fields are required!';
        $_SESSION['active_form'] = 'login';
        header("Location: index.php");
        exit();
    }

    // Fetch user (prepared statement)
    $stmt = $conn->prepare("SELECT name, email, password FROM users WHERE email = ?");
    $stmt->bind_param("s", $email);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows === 1) {
        $user = $result->fetch_assoc();

        if (password_verify($password, $user['password'])) {

            // Store session
            $_SESSION['name'] = $user['name'];
            $_SESSION['email'] = $user['email'];

            
            $username = urlencode($_SESSION['name']);
            header("Location: http://127.0.0.1:5000?user=$username");
            exit();
        }
    }

    // If login fails
    $_SESSION['login_error'] = 'Incorrect email or password';
    $_SESSION['active_form'] = 'login';
    header("Location: index.php");
    exit();
}
?>
