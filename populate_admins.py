from app import app, db, Admin
from werkzeug.security import generate_password_hash

FACULTY_ADMINS = {
    'admin': 'admin123',
    'harish': 'harish123',
    'rashmi': 'rashmi123',
    'faculty3': 'pass3',
    'faculty4': 'pass4',
    'faculty5': 'pass5',
    'faculty6': 'pass6',
    'faculty7': 'pass7',
    'faculty8': 'pass8',
    'faculty9': 'pass9',
    'faculty10': 'pass10',
    'faculty11': 'pass11'
}

with app.app_context():
    for username, password in FACULTY_ADMINS.items():
        admin = Admin.query.filter_by(username=username).first()
        if not admin:
            hashed_pw = generate_password_hash(password)
            new_admin = Admin(username=username, password=hashed_pw)
            db.session.add(new_admin)
            print(f"Added admin: {username}")
        else:
            print(f"Admin already exists: {username}")
    db.session.commit()
