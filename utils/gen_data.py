import numpy as np
import pandas as pd


def generate_synthetic_panel_data(n_countries=5, n_months=500):
    np.random.seed(42)
    dates = pd.date_range(start='2000-01-01', periods=n_months, freq='M')
    data = []
    for country in range(n_countries):
        default_rate = np.random.rand(n_months) * 5 + 1  # Random values for default rate
        macro_factor = np.random.rand(n_months) * 10 + 50  # Random macroeconomic factor
        country_data = pd.DataFrame({
            'date': dates,
            'country': f'Country_{country}',
            'default_rate': default_rate,
            'macro_factor': macro_factor
        })
        data.append(country_data)
    return pd.concat(data).reset_index(drop=True).assign(dfr_lag=lambda df: df.default_rate.shift(1)).dropna()

if __name__ == "__main__":
    data=generate_synthetic_panel_data();
    print(data)