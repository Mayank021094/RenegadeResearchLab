# ------------------Import Libraries ------------#
import pandas as pd
import sqlite3

# ---------------------CONSTANTS------------------#
db = sqlite3.connect("./raw_datasets/head_database.db")
cursor = db.cursor()
DB_user = sqlite3.connect("./raw_datasets/user_db.db")

# --------------------MAIN CODE-------------------#

# Creating Base Database For Stocks
stock_bucket = pd.read_csv(r'./raw_datasets/stock_buckets.csv')


variables = stock_bucket.columns
# Iterate over each column in the DataFrame
for i in variables:
    table_name = f"table_{i}"
    table_values = stock_bucket[i].dropna().tolist()  # Drop NaN values if any

    # Create the table if it does not exist
    if i == 'industry':
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({i} TEXT NOT NULL)")
    else:
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({i} TEXT UNIQUE NOT NULL)")

    # Insert values into the table
    for value in table_values:
        cursor.execute(f"INSERT OR IGNORE INTO {table_name} ({i}) VALUES (?)", (value,))

# Commit the transaction and close the connection
db.commit()

#Creating Nifty50 Database
nifty_50 = pd.read_csv('raw_datasets/nifty_50.csv')
nifty_50_symbols = nifty_50['symbol_nse'].tolist()

# Load head_database into a DataFrame
head_df = pd.read_sql("SELECT * FROM table_symbol_nse", db)

# Add a new column 'nifty_50' with 1 if symbol is in nifty_50, else 0
head_df['nifty_50'] = head_df['symbol_nse'].apply(lambda x: 1 if x in nifty_50_symbols else 0)

# Save the updated DataFrame back to the SQL database
head_df.to_sql('table_symbol_nse', db, if_exists='replace', index=False)

# Commit and close the database connection
db.commit()
db.close()

# Creating Database for Users
try:
    cursor = DB_user.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_db (
            id INTEGER PRIMARY KEY,
            fname TEXT NOT NULL,
            lname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            birth_date TEXT NOT NULL  -- Storing as TEXT in 'YYYY-MM-DD' format
        )
    """)

    DB_user.commit()

except Exception as e:
    print("An error occurred:", e)
    DB_user.rollback()  # Roll back any changes if an error occurs




finally:
    DB_user.close()  # Ensure the database connection is closed
