import pandas as pd
import numpy as np
import utils

# Create a dummy dataframe
dates = pd.date_range(start='2023-01-01', periods=8760, freq='H')
df = pd.DataFrame({'timestamp': dates, 'Load': np.random.rand(8760)*100, 'Solar': np.zeros(8760)})

# Test Case 1: Scaling ON
print("--- Test Case 1: Scaling ON ---")
utils.calculate_portfolio_metrics(
    df.copy(), 
    solar_capacity=100, 
    wind_capacity=0, 
    base_rec_price=8.00, 
    use_rec_scaling=True
)

# Test Case 2: Scaling OFF
print("\n--- Test Case 2: Scaling OFF ---")
res, df_res = utils.calculate_portfolio_metrics(
    df.copy(), 
    solar_capacity=100, 
    wind_capacity=0, 
    base_rec_price=8.00, 
    use_rec_scaling=False
)

# Check if REC Price is constant
unique_prices = df_res['REC_Price_USD'].unique()
print(f"Unique REC Prices (Should be [8.0]): {unique_prices}")
if len(unique_prices) == 1 and unique_prices[0] == 8.0:
    print("SUCCESS: REC Price is constant 8.00")
else:
    print("FAILURE: REC Price is not constant 8.00")
