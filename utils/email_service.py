import smtplib
from email.message import EmailMessage
import os

def send_otp_email(recipient_email, otp_code):
    sender_email = "abc387029@gmail.com"
    sender_password = "kaqowqwhvbcdqmdw"
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    msg = EmailMessage()
    msg['Subject'] = 'Your OTP for Registration'
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content(f'Your OTP code is: {otp_code}\n\nThis OTP will expire in 5 minutes.')

    try:
        # Note: If environment vars are dummy, it will fail here.
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send OTP email: {e}")
        return False

def send_result_email(recipient_email, student_name, result_url):
    sender_email = "abc387029@gmail.com"
    sender_password = "kaqowqwhvbcdqmdw"
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    msg = EmailMessage()
    msg['Subject'] = 'OMR Exam Result Available'
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    body = f"""Hello {student_name},

Your exam result has been generated.

Click the link below to view your result:

{result_url}

Regards,
OMR Evaluation System
"""
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send result email: {e}")
        return False
