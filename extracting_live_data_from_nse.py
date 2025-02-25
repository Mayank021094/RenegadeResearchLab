# ------------------Import Libraries ------------#
import pandas as pd
import sqlite3
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import lxml
import time
import datetime as dt
from book_values import BookValue
import random
from requests.exceptions import ConnectionError, Timeout, RequestException

# ---------------------CONSTANTS------------------#
URL = 'https://www.nseindia.com/companies-listing/corporate-filings-financial-results'
DB = sqlite3.connect("./raw_datasets/head_database.db")

# --------------------MAIN CODE-------------------#

# ------------SCRAPEING DATA FROM NSE WEBSITE------#
# Keep Chrome from closing
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('detach', True)

# Initialize the WebDriver with executable path
driver = webdriver.Chrome(options=chrome_options)
driver.get(URL)
time.sleep(10)

# Get the table
table_list = driver.find_element(By.TAG_NAME, 'table')
table_list_child = table_list.find_element(By.TAG_NAME, 'tbody')
row_list = table_list_child.find_elements(By.TAG_NAME, 'tr')

a_tags = [row.find_elements(By.TAG_NAME, 'a') for row in row_list]
td_tags = [row.find_elements(By.TAG_NAME, 'td') for row in row_list]

# Extracting XML links
href_list = []
for i in range(0, len(a_tags)):
    try:
        temp = a_tags[i][1].get_attribute('href')
    except:
        pass
    href_list.append(temp)
# print(len(href_list))
# print(href_list[1])

# Extract Consolidated/Non-Consolidated column
con_list = []
for i in range(0, len(td_tags)):
    try:
        temp = td_tags[i][3].text
    except:
        pass
    con_list.append(temp)
# print(len(con_list))
# print(con_list[1])

# Extract Company Name
company_list = []
for i in range(0, len(a_tags)):
    try:
        temp = a_tags[i][0].text
    except:
        pass
    company_list.append(temp)
# print(len(company_list))
# print(company_list[1])

# Closing the webdriver
driver.quit()

# Form a dataframe with the extracted value
financial_resuts = pd.DataFrame({'company_name': company_list,
                                 'consolidated_or_not': con_list,
                                 'xblr_links': href_list})

financial_resuts.drop_duplicates(inplace=True)
print(href_list[0])
# r = requests.get(href_list[0])
# html_file = r.text
# print(html_file)

# --------EXTRACTING DATA FROM XML FILES---------#
unique_companies = financial_resuts['company_name'].unique().tolist()

user_agent_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
]

object_dict = {}


# Retry logic function
def fetch_url_with_retry(url, headers, max_retries=3, backoff_factor=1):
    retries = 0
    while retries < max_retries:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            return r.text
        except (ConnectionError, Timeout, RequestException) as e:
            print(f"Error fetching {url}: {e}. Retrying... ({retries + 1}/{max_retries})")
            retries += 1
            time.sleep(backoff_factor * retries)  # Exponential backoff
    return None


for company in unique_companies:
    print(f"Processing company: {company}")

    temp_df = financial_resuts[financial_resuts['company_name'].str.lower().str.strip() == company.lower().strip()]
    print(f"Filtered DataFrame for {company}:")
    print(temp_df)
    if temp_df.empty:
        print(f"No data found for company: {company}")
        continue

    if len(temp_df) > 1:
        temp_df = temp_df[temp_df['consolidated_or_not'] != 'Non-Consolidated']
        print(f"After filtering consolidated data for {company}:")
        print(temp_df)

    if temp_df.empty:
        print(f"No consolidated data found for company: {company}")
        continue

    xbrl_link = temp_df['xblr_links'].values[0]
    print(f"xbrl link: {xbrl_link}")
    if len(temp_df) > 1:
        temp_df = temp_df[temp_df['consolidated_or_not'] != 'Non-Consolidated']

    user_agent = random.choice(user_agent_list)
    headers = {'User-Agent': user_agent}

    html_file = fetch_url_with_retry(xbrl_link, headers)

    if html_file:
        print('Success')
        obj = BookValue()
        soup = BeautifulSoup(html_file, 'xml')

        # Extract values with error handling
        obj.symbol = soup.find('xbrli:identifier').text if soup.find('xbrli:identifier') else None
        obj.end_date = soup.find('xbrli:endDate').text if soup.find('xbrli:endDate') else None
        obj.op_cashflow = float(soup.find("in-bse-fin:CashFlowsFromUsedInOperatingActivities").text) if soup.find(
            "in-bse-fin:CashFlowsFromUsedInOperatingActivities") else None
        obj.ppe = float(soup.find(
            "in-bse-fin:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities").text) if soup.find(
            "in-bse-fin:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities") else None
        obj.borr_current = float(soup.find('in-bse-fin:BorrowingsCurrent').text) if soup.find(
            'in-bse-fin:BorrowingsCurrent') else None
        obj.borr_noncurrent = float(soup.find('in-bse-fin:BorrowingsNoncurrent').text) if soup.find(
            'in-bse-fin:BorrowingsNoncurrent') else None
        obj.total_assets = float(soup.find('in-bse-fin:Assets').text) if soup.find('in-bse-fin:Assets') else None
        obj.total_liabilities = float(soup.find('in-bse-fin:Liabilities').text) if soup.find(
            'in-bse-fin:Liabilities') else None
        obj.paid_up_value = float(soup.find('in-bse-fin:PaidUpValueOfEquityShareCapital').text) if soup.find(
            'in-bse-fin:PaidUpValueOfEquityShareCapital') else None
        obj.face_value = float(soup.find('in-bse-fin:FaceValueOfEquityShareCapital').text) if soup.find(
            'in-bse-fin:FaceValueOfEquityShareCapital') else None
        obj.equity = float(soup.find('in-bse-fin:Equity').text) if soup.find('in-bse-fin:Equity') else None
        obj.pbt = float(soup.find('in-bse-fin:ProfitBeforeTax').text) if soup.find(
            'in-bse-fin:ProfitBeforeTax') else None
        obj.da = float(soup.find('in-bse-fin:DepreciationDepletionAndAmortisationExpense').text) if soup.find(
            'in-bse-fin:DepreciationDepletionAndAmortisationExpense') else None
        obj.fcost = float(soup.find('in-bse-fin:FinanceCosts').text) if soup.find('in-bse-fin:FinanceCosts') else None
        obj.earnings = float(soup.find('in-bse-fin:ProfitLossForPeriod').text) if soup.find(
            'in-bse-fin:ProfitLossForPeriod') else None
        obj.total_revenue = float(soup.find('in-bse-fin:Income').text) if soup.find('in-bse-fin:Income') else None
        obj.sales_from_ops = float(soup.find('in-bse-fin:RevenueFromOperations').text) if soup.find(
            'in-bse-fin:RevenueFromOperations') else None
        obj.dividend = float(soup.find('in-bse-fin:DividendsPaidClassifiedAsFinancingActivities').text) if soup.find(
            'in-bse-fin:DividendsPaidClassifiedAsFinancingActivities') else None
        obj.current_assets = float(soup.find('in-bse-fin:CurrentAssets').text) if soup.find(
            'in-bse-fin:CurrentAssets') else None
        obj.current_liabilities = float(soup.find('in-bse-fin:CurrentLiabilities').text) if soup.find(
            'in-bse-fin:CurrentLiabilities') else None

        # Store the object in the dictionary
        object_dict[company] = obj
    else:
        print(f"Failed to fetch data for {company} after retries.")


# ---------------------------SAVING THE DATA TO DATABASE----------------------------#

# Connect to database
db = sqlite3.connect('./raw_datasets/head_database.db')
cursor = db.cursor()

for company, obj in object_dict.items():
    # Match the symbol to the existing table_symbol_nse to find the index
    cursor.execute("SELECT symbol_nse FROM table_symbol_nse WHERE symbol_nse=?", (obj.symbol,))
    symbol_id = cursor.fetchone()

    if symbol_id is None:
        print(f"Symbol {obj.symbol} not found in database for {company}. Creating...")
        cursor.execute(f"INSERT OR IGNORE INTO table_symbol_nse (symbol_nse) VALUES (?)", (obj.symbol,))
        cursor.execute(f"INSERT OR IGNORE INTO table_stock (stock) VALUES (?)", (company,))
        yfinance_symbol = obj.symbol + '.NS'
        cursor.execute(f"INSERT OR IGNORE INTO table_symbol_yfinance (symbol_yfinance) VALUES (?)",
                       (yfinance_symbol,))
        cursor.execute("SELECT symbol_nse FROM table_symbol_nse WHERE symbol_nse=?", (obj.symbol,))
        symbol_id = cursor.fetchone()

    symbol = symbol_id[0]  # Use consistent variable naming

    # Create or update tables for each attribute of the object
    attributes = {
        'op_cashflow': obj.op_cashflow,
        'ppe': obj.ppe,
        'borr_current': obj.borr_current,
        'borr_noncurrent': obj.borr_noncurrent,
        'total_assets': obj.total_assets,
        'total_liabilities': obj.total_liabilities,
        'paid_up_value': obj.paid_up_value,
        'face_value': obj.face_value,
        'equity': obj.equity,
        'pbt': obj.pbt,
        'da': obj.da,
        'fcost': obj.fcost,
        'earnings': obj.earnings,
        'total_revenue': obj.total_revenue,
        'sales_from_ops': obj.sales_from_ops,
        'dividend': obj.dividend,
        'current_assets': obj.current_assets,
        'current_liabilities': obj.current_liabilities
    }

    for attribute, value in attributes.items():
        if value is not None:
            date = obj.end_date
            table_name = f"table_{attribute}"

            # Step 2a: Create the table if it doesn't exist, using symbol_nse as a unique key
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                symbol_nse TEXT UNIQUE NOT NULL
            );
            """
            cursor.execute(create_table_query)

            # Step 2b: Check if the column for the current end_date exists in the table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]

            if obj.end_date not in columns:
                # Add the end_date column if it doesn't exist
                alter_table_query = f"ALTER TABLE {table_name} ADD COLUMN [{obj.end_date}] REAL;"
                cursor.execute(alter_table_query)

            # Step 3: Insert or update the value in the end_date column for the symbol_nse
            # Use INSERT OR IGNORE to ensure that symbol_nse exists in the table
            insert_symbol_query = f"""
            INSERT OR IGNORE INTO {table_name} (symbol_nse) VALUES (?);
            """
            cursor.execute(insert_symbol_query, (symbol,))

            # Update the value in the end_date column for the symbol_nse
            update_query = f"""
            UPDATE {table_name} SET [{obj.end_date}] = ? WHERE symbol_nse = ?;
            """
            cursor.execute(update_query, (value, symbol))

# Commit changes at the end of the loop
db.commit()
db.close()