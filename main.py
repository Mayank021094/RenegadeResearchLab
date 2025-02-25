# ------------------ Import Libraries ---------------#
import pandas as pd
import sqlite3
import smtplib  # To send emails
from flask import session, Flask, render_template, request, url_for, redirect, flash, g
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin, login_user, LoginManager, login_required, current_user, logout_user
import os
import datetime as dt
from factors_momentum import Momentum
from factors_value import Value
from factors_quality import Quality
from optimization import Optimization

# --------------------- CONSTANTS ------------------#
app = Flask(__name__)
app.config['SECRET_KEY'] = "TEMP@0210"

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'log_in'


# Database helper function
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect("./raw_datasets/user_db.db")
        g.db.row_factory = sqlite3.Row  # This allows column access by name
    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()


# ------------------ User Loader ----------------#
@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM user_db WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    if user_data:
        user = User(id=user_data["id"], email=user_data["email"])
        return user
    return None


# ------------------- User Model ----------------#
class User(UserMixin):
    def __init__(self, id=None, email=None):
        self.id = id
        self.email = email


# -------------------- MAIN CODE -------------------#

# ---------------Login, Home, Signup---------------#

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login', methods=["GET", "POST"])
def log_in():
    db = get_db()
    cursor = db.cursor()
    if request.method == 'POST':
        # Retrieve data
        email = request.form.get('email')
        pwd = request.form.get('pwd')

        # Check if the email exists in the database
        cursor.execute("SELECT id, password FROM user_db WHERE email = ?", (email,))
        user_data = cursor.fetchone()

        # If email doesn't exist, flash an error message
        if not user_data:
            flash("Email does not exist. Please sign up first.")
            return redirect(url_for('log_in'))

        # Check if the password matches the stored hash
        user_id, hashed_pwd = user_data["id"], user_data["password"]
        if not check_password_hash(hashed_pwd, pwd):
            flash("Incorrect password. Please try again.")
            return redirect(url_for('log_in'))

        # Create a User instance, log in the user, and redirect to a protected route
        user = User(id=user_id, email=email)
        login_user(user)
        return redirect(url_for('dashboard'))

    return render_template("login.html")

@app.route('/sign_up', methods=["GET", "POST"])
def sign_up():
    db = get_db()
    cursor = db.cursor()
    if request.method == 'POST':
        # Retrieve form data
        email = request.form.get('email')
        fname = request.form.get('fname')
        lname = request.form.get('lname')
        dob = request.form.get('dob')
        pwd = request.form.get('pwd')
        confirm_pwd = request.form.get('confirmpwd')
        pno = request.form.get('pno')

        # Check if email already exists
        cursor.execute("SELECT 1 FROM user_db WHERE email = ?", (email,))
        if cursor.fetchone():
            flash("Email already exists. Please use a different email.")
            return redirect(url_for('sign_up'))

        # Check if phone number already exists
        cursor.execute("SELECT 1 FROM user_db WHERE phone = ?", (pno,))
        if cursor.fetchone():
            flash("Phone number already exists. Please use a different phone number.")
            return redirect(url_for('sign_up'))

        # Check if passwords match
        if pwd != confirm_pwd:
            flash("Passwords do not match. Please try again.")
            return redirect(url_for('sign_up'))

        # # Check if age is greater than 18
        # try:
        #     birth_date = dt.datetime.strptime(dob, "%m-%d-%Y")  # Assuming dob format is MM-DD-YYYY
        #     print(birth_date)
        #     age = (dt.datetime.now() - birth_date).days // 365
        #     if age < 18:
        #         flash("You must be at least 18 years old to sign up.")
        #         return redirect(url_for('sign_up'))
        # except ValueError:
        #     flash("Invalid date format. Please use MM-DD-YYYY for date of birth.")
        #     return redirect(url_for('sign_up'))

        # If all checks pass, hash the password and insert data into the database
        hashed_pwd = generate_password_hash(pwd, method='pbkdf2:sha256', salt_length=8)
        cursor.execute(
            "INSERT INTO user_db (fname, lname, email, password, phone, birth_date) VALUES (?, ?, ?, ?, ?, ?)",
            (fname, lname, email, hashed_pwd, pno, dob)
        )
        db.commit()

        flash("Registration successful!")
        return redirect(url_for('log_in'))

    return render_template("signup.html")


# -----------------------Navbar items---------------#
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template("dashboard.html")

#-----------------------Sidebar Items--------------#
#-----------------------Stocks---------------------#
@app.route('/stocks', methods=['GET', 'POST'])
def stocks():
    return render_template("stocks.html")

@app.route('/stocks/<string:id>', methods=['GET', 'POST'])
def strategy(id):
    if request.method == 'POST':
        strat = request.form.get('strategy')
        univ = request.form.get('asset')
        top_n = int(request.form.get('top_n'))  # Convert to integer
        parameters = {'strat': strat, 'universe': univ, 'top_n': top_n}
        if id == 'Momentum':
            wt_strategy = request.form.getlist('wt_strategy')
            print([strat, univ, wt_strategy])
            parameters['wt'] = wt_strategy
            parameters['No. of Strategies selected'] = len(wt_strategy)
            strat_func = Momentum(strat=strat, univ=univ, wt_strat=wt_strategy)
        elif id == 'Value':
            wt_strategy = request.form.get('wt_strategy')
            parameters['wt'] = wt_strategy
            print(parameters)
            strat_func = Value(strat=strat, univ=univ, wt_strat=wt_strategy)
        elif id == 'Quality':
            wt_strategy = request.form.get('wt_strategy')
            print([strat, univ, wt_strategy])
            parameters['wt'] = wt_strategy
            strat_func = Quality(strat=strat, univ=univ, wt_strat=wt_strategy)
        elif id == 'Optimization':
            strat_func = Optimization(strat=strat, univ=univ)

        wts = strat_func.get_wts()
        session['param'] = parameters
        session['wts'] = wts.to_json()  # Convert DataFrame to JSON string
        return redirect(url_for('results', id=id))
    return render_template("strategy_body.html", id=id)

@app.route('/stocks/<string:id>/Results', methods=['GET', 'POST'])
def results(id):
    wts_json = session.get('wts')
    param = session.get('param')
    wts = pd.read_json(wts_json)  # Convert JSON string back to DataFrame
    return render_template('results_stocks.html', id=id, wts=wts, param=param)


if __name__ == "__main__":
    app.run(debug=True)
