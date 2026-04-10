import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super_secret_key_void_fill'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///omr_system.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "connect_args": {
            "timeout": 20
        }
    }
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    PROCESSED_FOLDER = os.path.join('static', 'processed')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024 # 100 MB limits
    # Email configuration
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
