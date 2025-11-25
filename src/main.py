import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# CONSTANTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DB_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, 'reservoir.db')

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'illinois.txt')
FIGURES_DIR = os.path.join(PROJECT_ROOT,'data', 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


DREDGING_COST = 5.00 # USD per ton

# most likely to vary for diff datasets/rivers
def ingest_and_clean():
    """Reads raw text, cleans columns, returns DataFrame."""
    print("Step 1: Ingesting Data...")
    df_raw = pd.read_csv(RAW_DATA_PATH, sep='\t', header = 0, comment='#')
    df_raw = df_raw.drop(0).copy()
    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

    cols_to_convert = df_raw.columns[3:]
    for col in cols_to_convert:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    df_raw.drop(columns=['agency_cd', '49313_80155_00003', '49314_00065_00003', 'site_no','49312_80154_00003_cd', '49313_80155_00003_cd', '49314_00065_00003_cd', '49315_00060_00003_cd'], inplace=True)
    df = df_raw.loc[:, [
        'datetime', '49312_80154_00003', '49315_00060_00003']].rename(
            columns={'datetime':'Date',
                     '49312_80154_00003': 'ss_mgpl',
                     '49315_00060_00003': 'flow_cfs'}).set_index('Date')

    return df




def impute_missing_data(df):
    """Uses Log-Log Regression to fill missing sediment values."""
    print("Step 2: Imputing Missing Values...")

    X = np.log(df['flow_cfs'])
    y = np.log(df['ss_mgpl'])

    plt.figure(figsize=(10,6))
    plt.scatter(X,y, alpha=0.1)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title('Sediment Rating Curve')
    plt.savefig(os.path.join(FIGURES_DIR, 'sediment_rating_curve.png'))
    print(f"Sediment Rating Curve saved at {os.path.join(FIGURES_DIR, 'sediment_rating_curve.png')}.")

    dfc = df.dropna().copy()
    dftrain, dftest = train_test_split(dfc, test_size = 0.2, random_state = 42)

    X_train = np.log(dftrain['flow_cfs']).values.reshape(-1,1)
    X_test = np.log(dftest['flow_cfs']).values.reshape(-1,1)
    y_train = np.log(dftrain['ss_mgpl'])
    y_test = np.log(dftest['ss_mgpl'])

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    error = np.exp(rmse)

    # retraining on full data
    fmodel = LinearRegression()
    X_full = np.log(df.dropna()['flow_cfs']).values.reshape(-1, 1)
    y_full = np.log(df.dropna()['ss_mgpl'])
    fmodel.fit(X_full, y_full)
    mask_missing = df['ss_mgpl'].isnull()
    X_missing = np.log(df.loc[mask_missing, 'flow_cfs']).values.reshape(-1, 1)
    y_missing_pred = fmodel.predict(X_missing)
    df.loc[mask_missing, 'ss_mgpl'] = np.exp(y_missing_pred)
    print(f"Imputation complete with RMSE: {rmse}, Error Factor: {error}")
    return df


def save_to_vault(df):
    """Saves clean data to SQLite."""
    print(f"Step 3: Saving to Vault at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('daily_sediment', conn, if_exists='replace', index=True)
    conn.close()
    print("Data saved to Vault.")




def generate_financial_report():
    """Reads from Vault, calculates Liability, prints Summary."""
    print("Step 4: Generating Financial Report...")

    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT date, flow_cfs, ss_mgpl
        FROM daily_sediment
        ORDER by date
    """
    df_viz = pd.read_sql_query(query, conn, parse_dates=['Date'])
    conn.close()

    df_viz['daily_load_tons'] = df_viz['ss_mgpl'] * df_viz['flow_cfs'] * 0.0027
    df_viz['cumulative_sediment_tons'] = df_viz['daily_load_tons'].cumsum()

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_phys = 'tab:brown'
    ax1.set_xlabel('Date (Year)', fontsize=12)
    ax1.set_ylabel('Cumulative Sediment (Millions of Tons)', color=color_phys, fontsize=12)
    ax1.plot(df_viz['Date'], df_viz['cumulative_sediment_tons'] / 1_000_000, color=color_phys, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_phys)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_fin = 'tab:green'
    ax2.set_ylabel('Cumulative Dredging Liability (USD Millions)', color=color_fin, fontsize=12)

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min * DREDGING_COST, y_max * DREDGING_COST)
    ax2.tick_params(axis='y', labelcolor=color_fin)
    formatter = ticker.FormatStrFormatter('$%.1f M')
    ax2.yaxis.set_major_formatter(formatter)
    plt.title('The Death Curve: Cumulative Sediment Load & Financial Liability', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'death_curve.png'))
    print(f"Death Curve saved at {os.path.join(FIGURES_DIR, 'death_curve.png')}.")


    total_cost = df_viz['cumulative_sediment_tons'].iloc[-1] * (DREDGING_COST / 1_000_000)
    print(f"Total Liability: ${total_cost:.2f} Million")

if __name__ == "__main__":
    df = ingest_and_clean()
    df = impute_missing_data(df)
    save_to_vault(df)
    generate_financial_report()
    print("Mission Complete.")
