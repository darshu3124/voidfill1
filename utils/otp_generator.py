import secrets
import string

def generate_otp(length=6):
    """Generates a secure random OTP of the given length."""
    alphabet = string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))
