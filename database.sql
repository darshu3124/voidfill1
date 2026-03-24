-- Create Database
CREATE DATABASE IF NOT EXISTS omr_system;
USE omr_system;

-- Admin Table
CREATE TABLE IF NOT EXISTS admin (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

-- Students Table
CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OTP Table
CREATE TABLE IF NOT EXISTS email_otps (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(150) NOT NULL,
    otp_code VARCHAR(6) NOT NULL,
    expires_at DATETIME NOT NULL
);

-- Answer Key Table
CREATE TABLE IF NOT EXISTS answer_key (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question_number INT NOT NULL UNIQUE,
    correct_option CHAR(1) NOT NULL
);

-- Results Table
CREATE TABLE IF NOT EXISTS results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT,
    subject_id INT,
    student_email VARCHAR(150),
    marks INT,
    total_marks INT,
    percentage FLOAT,
    result_pdf VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES subject(id) ON DELETE CASCADE
);

-- Insert Default Admin
-- Password is 'admin123' hashed using pbkdf2:sha256
INSERT IGNORE INTO admin (username, password) 
VALUES ('admin', 'scrypt:32768:8:1$kF9B9Kx6I8L8qgWf$91444fb9d2a0d187796d8e2003c2a0d0a52cd1d37b6de3d12d4d8c83a1528652');
