import json
from app import app, db, Student
from werkzeug.security import generate_password_hash

def import_students(json_file):
    try:
        with open(json_file, 'r') as f:
            students_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please create it first.")
        return
    except json.JSONDecodeError:
        print(f"Error: {json_file} contains invalid JSON.")
        return

    with app.app_context():
        count_added = 0
        count_updated = 0
        
        for data in students_data:
            username = data.get('username')
            password = data.get('password')
            name = data.get('name', '') # Leave blank instead of fallback to username if name is missing
            
            if not username or not password:
                print(f"Skipping entry missing username or password: {data}")
                continue

            # We use the existing 'email' column to store the 'username' 
            # as it is a unique identifier required by the database.
            student = Student.query.filter_by(email=username).first()
            
            if student:
                # Update existing student's password and name
                student.password = generate_password_hash(password)
                student.name = name
                student.verified = True # Skip email verification
                count_updated += 1
            else:
                # Create a new student
                hashed_pw = generate_password_hash(password)
                new_student = Student(name=name, email=username, password=hashed_pw, verified=True)
                db.session.add(new_student)
                count_added += 1
                
        db.session.commit()
        print(f"Successfully added {count_added} and updated {count_updated} students.")

if __name__ == '__main__':
    # You can change the filename if your JSON file is named differently
    import_students('students.json')
