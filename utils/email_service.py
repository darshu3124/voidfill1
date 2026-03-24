import smtplib
from email.message import EmailMessage
import os
import threading

def _send_email_async(msg, smtp_server, smtp_port, sender_email, sender_password):
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true':
                server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        print(f"Failed to send email async: {e}")

def send_otp_email(recipient_email, otp_code):
    sender_email = os.environ.get('MAIL_USERNAME')
    sender_password = os.environ.get('MAIL_PASSWORD')
    smtp_server = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('MAIL_PORT', 587))

    if not sender_email or not sender_password:
        print("Mail server/credentials not configured in .env")
        return False

    msg = EmailMessage()
    msg['Subject'] = 'Your OTP for Registration'
    msg['From'] = f"VoidFill <{sender_email}>"
    msg['To'] = recipient_email
    msg.set_content(f'Your OTP code is: {otp_code}\n\nThis OTP will expire in 5 minutes.')

    threading.Thread(target=_send_email_async, args=(msg, smtp_server, smtp_port, sender_email, sender_password)).start()
    return True

def send_result_email(recipient_email, student_name, result_url):
    sender_email = os.environ.get('MAIL_USERNAME')
    sender_password = os.environ.get('MAIL_PASSWORD')
    smtp_server = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('MAIL_PORT', 587))

    if not sender_email or not sender_password:
        return False

    msg = EmailMessage()
    msg['Subject'] = 'OMR Exam Result Available'
    msg['From'] = f"VoidFill <{sender_email}>"
    msg['To'] = recipient_email
    
    body = f"""Hello {student_name},

Your exam result has been generated.

Click the link below to view your result:

{result_url}

Regards,
VoidFill OMR System
"""
    msg.set_content(body)

    threading.Thread(target=_send_email_async, args=(msg, smtp_server, smtp_port, sender_email, sender_password)).start()
    return True
