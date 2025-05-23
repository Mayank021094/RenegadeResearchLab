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
from options import option_json
import pickle
import json
from figures_and_tables import edge_table, estimator_table, plot_volatility_cone_plotly, plot_payoff_chart
from plotly.io import to_html

# --------------------- CONSTANTS ------------------#
app = Flask(__name__)
app.config['SECRET_KEY'] = "TEMP@0210"
app.config['LOGIN_DISABLED'] = False

# Initialize LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'log_in'  # This route is used when a user is not authenticated


# ------------------ Database Helper ----------------#
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect("./raw_datasets/user_db.db")
        g.db.row_factory = sqlite3.Row  # Allows column access by name
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


# ------------------ User Model ----------------#
class User(UserMixin):
    def __init__(self, id=None, email=None):
        self.id = id
        self.email = email


# ------------------ MAIN CODE ----------------#

# --------------- Home, Login, Signup Routes ---------------#
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/login', methods=["GET", "POST"])
def log_in():
    db = get_db()
    cursor = db.cursor()
    if request.method == 'POST':
        # Retrieve form data
        email = request.form.get('email')
        pwd = request.form.get('pwd')

        # Check if the email exists in the database
        cursor.execute("SELECT id, password FROM user_db WHERE email = ?", (email,))
        user_data = cursor.fetchone()

        if not user_data:
            flash("Email does not exist. Please sign up first.")
            return redirect(url_for('log_in'))

        # Verify the password
        user_id, hashed_pwd = user_data["id"], user_data["password"]
        if not check_password_hash(hashed_pwd, pwd):
            flash("Incorrect password. Please try again.")
            return redirect(url_for('log_in'))

        # Create a user instance and log them in
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

        # Ensure passwords match
        if pwd != confirm_pwd:
            flash("Passwords do not match. Please try again.")
            return redirect(url_for('sign_up'))

        # Hash the password and insert new user data into the database
        hashed_pwd = generate_password_hash(pwd, method='pbkdf2:sha256', salt_length=8)
        cursor.execute(
            "INSERT INTO user_db (fname, lname, email, password, phone, birth_date) VALUES (?, ?, ?, ?, ?, ?)",
            (fname, lname, email, hashed_pwd, pno, dob)
        )
        db.commit()

        flash("Registration successful!")
        return redirect(url_for('log_in'))

    return render_template("signup.html")


# ------------------ Protected Routes (Require Login) ----------------#

# Dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template("dashboard.html")


# -------------------------------------------------------------
#                         Stocks
# -------------------------------------------------------------
@app.route('/stocks', methods=['GET', 'POST'])
@login_required
def stocks():
    return render_template("stocks.html")


# Stock Strategy Page
@app.route('/stocks/<string:id>', methods=['GET', 'POST'])
@login_required
def strategy(id):
    if request.method == 'POST':
        strat = request.form.get('strategy')
        univ = request.form.get('asset')
        top_n = int(request.form.get('top_n'))  # Convert to integer
        parameters = {'strat': strat, 'universe': univ, 'top_n': top_n}
        if id == 'Momentum':
            wt_strategy = request.form.getlist('wt_strategy')
            parameters['wt'] = wt_strategy
            parameters['No. of Strategies selected'] = len(wt_strategy)
            strat_func = Momentum(strat=strat, univ=univ, wt_strat=wt_strategy)
        elif id == 'Value':
            wt_strategy = request.form.get('wt_strategy')
            parameters['wt'] = wt_strategy
            strat_func = Value(strat=strat, univ=univ, wt_strat=wt_strategy)
        elif id == 'Quality':
            wt_strategy = request.form.get('wt_strategy')
            parameters['wt'] = wt_strategy
            strat_func = Quality(strat=strat, univ=univ, wt_strat=wt_strategy)
        elif id == 'Optimization':
            strat_func = Optimization(strat=strat, univ=univ)

        wts = strat_func.get_wts()
        session['param'] = parameters
        session['wts'] = wts.to_json()  # Convert DataFrame to JSON string
        return redirect(url_for('results', id=id))
    return render_template("strategy_body.html", id=id)


# Stocks Results Page
@app.route('/stocks/<string:id>/Results', methods=['GET', 'POST'])
@login_required
def results(id):
    wts_json = session.get('wts')
    param = session.get('param')
    wts = pd.read_json(wts_json)  # Convert JSON string back to DataFrame
    return render_template('results_stocks.html', id=id, wts=wts, param=param)


# -------------------------------------------------------------
#                         Options
# -------------------------------------------------------------
# ---------------------- Global Caching for Options Data ----------------------
# (Note: For testing purposes only. In production, consider a proper caching mechanism.)
_cached_options_json = None
_cached_edge_df = None
_kite_obj = None


# ---------------------- Options Routes ----------------------
@app.route('/options', methods=['GET', 'POST'])
@login_required
def options_generate_session():
    return render_template("options_generate_session.html")


@app.route('/options/dashboard', methods=['GET', 'POST'])
@login_required
def options():
    if request.method == 'POST':
        api_key = request.form.get('api_key', '').strip()
        secret_key = request.form.get('secret_key', '').strip()
        client_id = request.form.get('client_id', '').strip()

    global _cached_options_json, _cached_edge_df, _kite_obj

    if _cached_options_json is None:
        _cached_options_json, _kite_obj = option_json(api_key, secret_key, client_id)
        # with open("options_json_for_ui_testing_2.pkl", "rb") as file:
        #     _cached_options_json = pickle.load(file)
        # Note: Do not convert datetime objects to strings here;
        # we need them in their original format for further calculations.
        _cached_edge_df = edge_table(_cached_options_json)
    edge_data = list(_cached_edge_df.to_dict(orient="records"))
    enumerated_edge_data = list(enumerate(edge_data, start=1))
    return render_template("options.html", enumerated_edge_data=enumerated_edge_data)

@app.route('/options/<string:ticker>/<expiry>', methods=['GET', 'POST'])
@login_required
def option_analysis(ticker, expiry):
    global _cached_options_json  # Ensure we are accessing the global variable
    # Convert expiry (a string) back to a datetime object if needed.
    expiry_date = dt.datetime.strptime(expiry, '%Y-%m-%d').date()
    # Use the cached options_json (which still has datetime objects intact)
    # If _cached_options_json is None, load it from the pickle file.
    if _cached_options_json is None:
        with open("options_json_for_ui_testing_2.pkl", "rb") as file:
            _cached_options_json = pickle.load(file)
    options_json = _cached_options_json
    symbols_json = options_json[ticker]
    est_df = estimator_table(symbols_json, expiry_date)
    est_data = list(est_df.to_dict(orient="records"))
    realized_vol_json = symbols_json['realized_volatility']
    vol_cone = plot_volatility_cone_plotly(cones_df=realized_vol_json['Volatility_Cones_df'])
    skew_cone = plot_volatility_cone_plotly(cones_df=realized_vol_json['Skewness_Cones_df'], moment='Skewness')
    kurt_cone = plot_volatility_cone_plotly(cones_df=realized_vol_json['Kurtosis_Cones_df'], moment='Kurtosis')

    vol_cone.update_layout(
        width=1500,  # explicit width
        height=1000,  # explicit height,
        margin=dict(l=100, r=200, t=100, b=100),
    )
    vol_cone_html = to_html(vol_cone, full_html=False, include_plotlyjs='cdn', config={'responsive': False})

    skew_cone.update_layout(
        width=1500,  # explicit width
        height=1000,  # explicit height,
        margin=dict(l=100, r=200, t=100, b=100),
    )
    skew_cone_html = to_html(skew_cone, full_html=False, include_plotlyjs=False, config={'responsive': False})

    kurt_cone.update_layout(
        width=1500,  # explicit width
        height=1000,  # explicit height,
        margin=dict(l=100, r=200, t=100, b=100),
    )
    kurt_cone_html = to_html(kurt_cone, full_html=False, include_plotlyjs=False, config={'responsive': False})

    return render_template(
        "option_analysis.html",
        ticker=ticker,
        expiry=expiry_date,
        est_data=est_data,
        vol_cone=vol_cone_html,
        skew_cone=skew_cone_html,
        kurt_cone=kurt_cone_html
    )


@app.route('/options/<string:ticker>/<expiry>/strategies/', methods=['GET', 'POST'])
@login_required
def options_strategy_analysis(ticker, expiry):
    global _cached_options_json

    # Validate the expiry date format.
    try:
        expiry_date = dt.datetime.strptime(expiry, '%Y-%m-%d').date()
    except ValueError as e:
        flash("Invalid expiry date format. Expected YYYY-MM-DD.", "error")
        return redirect(url_for('error_page'))  # Replace 'error_page' with your error route.

    # Try loading the cached JSON if not already loaded.
    try:
        if _cached_options_json is None:
            # _cached_options_json = option_json()  # Uncomment if calculating on the fly.
            with open("options_json_for_ui_testing_2.pkl", "rb") as file:
                _cached_options_json = pickle.load(file)
    except (IOError, pickle.UnpicklingError) as e:
        flash("Error loading options data: " + str(e), "error")
        return redirect(url_for('error_page'))

    options_json = _cached_options_json

    # Check if the ticker exists in the options JSON.
    if ticker not in options_json:
        flash(f"Ticker '{ticker}' not found in data.", "error")
        return redirect(url_for('error_page'))

    symbols_json = options_json[ticker]

    # Check if strategies exist for the given expiry.
    if 'strategies' not in symbols_json or expiry_date not in symbols_json['strategies']:
        flash(f"No strategies available for expiry: {expiry_date}.", "error")
        return redirect(url_for('error_page'))

    strategies_json = symbols_json['strategies'][expiry_date]
    strategies = list(strategies_json.keys())
    output_json = {}

    for strat in strategies:
        # Retrieve the greeks; skip strategy if missing.
        try:
            greeks = strategies_json[strat]['greeks']
        except Exception as e:
            app.logger.error(f"Missing greeks for strategy {strat}: {str(e)}")
            continue

        # Retrieve the strike value; default to 0 if not found.
        K = greeks.get('Strike', 0)

        # Retrieve the payoff DataFrame; skip if not found.
        try:
            df = strategies_json[strat]['payoff']['payoffs']
        except Exception as e:
            app.logger.error(f"Missing payoff data for strategy {strat}: {str(e)}")
            continue

        # Generate the payoff chart and convert it to an HTML snippet.
        try:
            fig = plot_payoff_chart(df=df, K=K, title=strat)
            fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
        except Exception as e:
            app.logger.error(f"Error generating plot for strategy {strat}: {str(e)}")
            continue

        output_json[strat] = {"greeks": greeks, "figure": fig_html}

    if not output_json:
        flash("No valid strategies found.", "warning")
        return redirect(url_for('error_page'))

    return render_template("options_strategies.html",
                           ticker=ticker,
                           expiry=expiry,
                           strategies=output_json)

if __name__ == "__main__":
    app.run(debug=True)
