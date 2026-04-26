from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import re
import random
import string

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MySQL Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',         # Default is 'root'
    'password': 'berrybeshe',         # Enter your MySQL password here
    'database': 'login_db'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# Captcha Generator
def generate_captcha():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

# Password Strength Validation
def is_password_strong(password):
    if len(password) < 8: return False
    if not re.search("[a-z]", password): return False
    if not re.search("[A-Z]", password): return False
    if not re.search("[0-9]", password): return False
    if not re.search("[_@$!%*#?&]", password): return False
    return True

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        user_captcha = request.form['captcha']

        if user_captcha != session.get('captcha_reg'):
            flash("Invalid Captcha!", "error")
        elif not is_password_strong(password):
            flash("Password must be 8+ chars with Uppercase, Lowercase, Number & Symbol.", "error")
        else:
            hashed_pw = generate_password_hash(password)
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                # Note: MySQL uses %s instead of ?
                cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, hashed_pw))
                conn.commit()
                cursor.close()
                conn.close()
                flash("Registration Successful! Please Login.", "success")
                return redirect(url_for('login'))
            except mysql.connector.Error as err:
                flash(f"Error: Email might already exist.", "error")

    session['captcha_reg'] = generate_captcha()
    return render_template('register.html', captcha=session['captcha_reg'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_captcha = request.form['captcha']

        if user_captcha != session.get('captcha_login'):
            flash("Invalid Captcha!", "error")
        else:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()

            if user and check_password_hash(user['password'], password):
                session['user'] = user['name']
                return redirect(url_for('index'))
            else:
                flash("Invalid Email or Password.", "error")

    session['captcha_login'] = generate_captcha()
    return render_template('login.html', captcha=session['captcha_login'])

@app.route('/index')
def index():
    if 'user' in session:
        return render_template('index.html', name=session['user'])
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
