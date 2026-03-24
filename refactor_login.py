import re

def refactor():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Replace url_for arguments universally
    content = content.replace("url_for('admin_login')", "url_for('login')")
    content = content.replace("url_for('student_login')", "url_for('login')")
    content = content.replace("url_for('student_login', next=request.url)", "url_for('login', next=request.url)")
    content = content.replace("login_manager.login_view = 'student_login'", "login_manager.login_view = 'login'")

    # 2. Remove old admin_login function
    admin_login_pattern = re.compile(r"@app\.route\('/admin_login'.*?return render_template\('admin_login\.html'\)", re.DOTALL)
    content = admin_login_pattern.sub('', content)

    # 3. Remove old student_login function
    student_login_pattern = re.compile(r"@app\.route\('/student_login'.*?return render_template\('student_login\.html'\)", re.DOTALL)
    
    new_login_route = """@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 1. Check if Admin
        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password, password):
            session['admin_id'] = admin.id
            session['role'] = 'admin'
            flash('Logged in successfully as Admin.', 'success')
            return redirect(url_for('admin_dashboard'))
            
        # 2. Check if Student
        student = Student.query.filter((Student.name == username) | (Student.email == username)).first()
        if student and check_password_hash(student.password, password):
            if not student.verified:
                flash('Please verify your email first.', 'warning')
                return redirect(url_for('login'))
                
            login_user(student)
            session['role'] = 'student'
            session['student_id'] = student.id
            flash('Logged in successfully.', 'success')
            
            next_url = request.form.get('next') or request.args.get('next')
            if next_url:
                from urllib.parse import urlparse
                if not urlparse(next_url).netloc:
                    return redirect(next_url)
                    
            return redirect(url_for('student_dashboard'))
            
        flash('Invalid username/email or password', 'danger')
        
    return render_template('login.html')
"""
    content = student_login_pattern.sub(new_login_route, content)

    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    refactor()
