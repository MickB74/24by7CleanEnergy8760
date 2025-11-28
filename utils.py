import pandas as pd
import numpy as np
import io
import json
import zipfile
import random

# Typical Hourly Load Profiles (% of Peak Load)
LOAD_PROFILES = {
    'Office':      [10, 8, 7, 6, 7, 10, 25, 45, 70, 85, 95, 100, 100, 95, 90, 90, 85, 80, 65, 45, 30, 20, 15, 12],
    'Data Center': [95, 95, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99, 99, 99, 99, 98, 98, 98, 98, 97, 97, 96, 96, 95],
    'Retail':      [5, 4, 3, 3, 4, 6, 10, 25, 50, 75, 85, 95, 100, 100, 95, 90, 85, 80, 75, 60, 40, 25, 15, 8],
    'Residential': [35, 30, 25, 25, 30, 45, 70, 85, 80, 75, 70, 65, 65, 70, 75, 80, 85, 90, 100, 95, 80, 60, 50, 40],
    'Hospital':    [85, 85, 85, 85, 86, 87, 88, 90, 92, 95, 95, 96, 96, 97, 97, 97, 96, 95, 95, 94, 93, 90, 88, 86],
    'Warehouse':   [15, 12, 10, 10, 12, 20, 35, 50, 65, 75, 85, 90, 90, 85, 80, 80, 70, 60, 50, 40, 30, 25, 20, 18]
}

def generate_load_profile_shape(dates, building_type):
    """
    Generates a normalized load profile shape for a given building type.
    Uses defined 24-hour profiles with added seasonality and noise.
    """
    day_of_year = dates.dayofyear.to_numpy()
    hour_of_day = dates.hour.to_numpy()
    
    # Get base 24-hour profile
    base_profile_24h = np.array(LOAD_PROFILES.get(building_type, LOAD_PROFILES['Office']))
    
    # Map to full year
    # hour_of_day is 0-23, so we can directly index
    profile = base_profile_24h[hour_of_day]
    
    # Add Seasonality (Summer Peak)
    # Peak at day 200 (mid-July), min at day 15 (mid-Jan)
    # Amplitude depends on building type? Let's keep it simple for now.
    # Residential might have higher seasonality (AC/Heating).
    # Data Center might have less.
    
    if building_type == 'Data Center':
        seasonality = 1.0 + 0.05 * np.cos((day_of_year - 200) * 2 * np.pi / 365)
    elif building_type == 'Residential':
        seasonality = 1.0 + 0.4 * np.cos((day_of_year - 200) * 2 * np.pi / 365)
    else:
        seasonality = 1.0 + 0.2 * np.cos((day_of_year - 200) * 2 * np.pi / 365)
        
    # Apply seasonality
    profile = profile * seasonality
    
    # Add Random Noise
    noise = np.random.normal(0, 2, size=len(dates)) # +/- 2% noise
    profile = profile + noise
    
    return np.maximum(profile, 0)

# eGRID 2023 Output Emission Rates (lb CO2e/MWh)
EGRID_FACTORS = {
    "National Average": 820.0,
    "ERCOT": 733.9,
    "CAISO": 428.5,
    "ISO-NE": 633.0,
    "SPP": 867.0,
    "MISO": 747.4,
    "NYISO": 230.0, # Estimate based on clean grid
    "PJM": 800.0    # Estimate
}

REGIONAL_PARAMS = {
    "National Average": {
        "solar_seasonality": 0.4, "solar_cloud": 0.5,
        "wind_seasonality": 0.2, "wind_daily_amp": 0.3, "wind_peak_hour": 4, "wind_base": 30
    },
    "ERCOT": {
        "solar_seasonality": 0.5, "solar_cloud": 0.3, # Sunny, hot summers
        "wind_seasonality": 0.3, "wind_daily_amp": 0.5, "wind_peak_hour": 2, "wind_base": 35 # Strong night wind
    },
    "CAISO": {
        "solar_seasonality": 0.6, "solar_cloud": 0.2, # Very sunny
        "wind_seasonality": 0.2, "wind_daily_amp": 0.4, "wind_peak_hour": 18, "wind_base": 25 # Evening wind (sea breeze)
    },
    "PJM": {
        "solar_seasonality": 0.5, "solar_cloud": 0.6, # Cloudier
        "wind_seasonality": 0.4, "wind_daily_amp": 0.2, "wind_peak_hour": 14, "wind_base": 28 # Winter peak
    },
    "NYISO": {
        "solar_seasonality": 0.5, "solar_cloud": 0.6,
        "wind_seasonality": 0.4, "wind_daily_amp": 0.2, "wind_peak_hour": 14, "wind_base": 28
    },
    "ISO-NE": {
        "solar_seasonality": 0.5, "solar_cloud": 0.6,
        "wind_seasonality": 0.4, "wind_daily_amp": 0.2, "wind_peak_hour": 14, "wind_base": 28
    },
    "MISO": {
        "solar_seasonality": 0.45, "solar_cloud": 0.5,
        "wind_seasonality": 0.3, "wind_daily_amp": 0.4, "wind_peak_hour": 3, "wind_base": 38 # Strong night wind
    },
    "SPP": {
        "solar_seasonality": 0.45, "solar_cloud": 0.4,
        "wind_seasonality": 0.3, "wind_daily_amp": 0.4, "wind_peak_hour": 3, "wind_base": 40 # Very strong wind
    }
}

def generate_synthetic_8760_data(year=2023, building_portfolio=None, region="National Average"):
    """
    Generates synthetic 8760 hourly data for Solar, Wind, and Load.
    building_portfolio: List of dicts [{'type': 'Office', 'annual_mwh': 1000}, ...]
    region: String, one of the keys in REGIONAL_PARAMS
    Returns a DataFrame with datetime index and columns: 'Solar', 'Wind', 'Load' (Total), plus individual building loads.
    """
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:00:00', freq='h')
    
    params = REGIONAL_PARAMS.get(region, REGIONAL_PARAMS["National Average"])
    
    # Solar: Peak in summer, zero at night, bell curve during day
    day_of_year = dates.dayofyear.to_numpy()
    hour_of_day = dates.hour.to_numpy()
    
    # Seasonality (peak in summer)
    seasonality = 1 + params["solar_seasonality"] * np.cos((day_of_year - 172) * 2 * np.pi / 365)
    
    # Daily pattern (0 at night, peak at noon)
    daily_pattern = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12))
    daily_pattern[hour_of_day < 6] = 0
    daily_pattern[hour_of_day > 18] = 0
    
    solar_profile = seasonality * daily_pattern * 100 
    cloud_cover = np.random.beta(2, 5, size=len(dates))
    solar_profile = solar_profile * (1 - cloud_cover * params["solar_cloud"])
    
    # Wind
    wind_seasonality = 1 + params["wind_seasonality"] * np.cos((day_of_year - 15) * 2 * np.pi / 365) 
    wind_daily = 1 + params["wind_daily_amp"] * np.cos((hour_of_day - params["wind_peak_hour"]) * 2 * np.pi / 24) 
    wind_noise = np.random.weibull(2, size=len(dates))
    wind_profile = wind_seasonality * wind_daily * wind_noise * params["wind_base"] 
    wind_profile = np.clip(wind_profile, 0, 100) 
    
    # Nuclear - Baseload with 90-95% capacity factor (constant output)
    nuclear_capacity_factor = 0.92 + np.random.normal(0, 0.01, size=len(dates))
    nuclear_capacity_factor = np.clip(nuclear_capacity_factor, 0.88, 0.96)
    nuclear_profile = nuclear_capacity_factor * 100
    
    # Geothermal - Baseload with regional variation
    # Higher in Western US (CAISO, SPP), lower elsewhere
    geo_base_cf = 0.85 if region in ["CAISO", "SPP"] else 0.75
    geo_capacity_factor = geo_base_cf + np.random.normal(0, 0.02, size=len(dates))
    geo_capacity_factor = np.clip(geo_capacity_factor, geo_base_cf - 0.05, geo_base_cf + 0.05)
    geothermal_profile = geo_capacity_factor * 100
    
    # Hydropower - Seasonal variation (higher in spring/summer, lower in fall/winter)
    # Regional variation in capacity factor
    hydro_base_cf = 0.45 if region == "CAISO" else (0.40 if region in ["PJM", "NYISO", "ISO-NE"] else 0.35)
    # Spring runoff pattern: peak in April-June, low in fall/winter
    hydro_seasonality = 1 + 0.4 * np.sin((day_of_year - 120) * 2 * np.pi / 365)  # Peak around day 120 (late April)
    hydro_capacity_factor = hydro_base_cf * hydro_seasonality
    # Add some daily variation (water flow management)
    hydro_daily_pattern = 1 + 0.1 * np.sin((hour_of_day - 12) * 2 * np.pi / 24)
    hydro_profile = hydro_capacity_factor * hydro_daily_pattern * 100
    hydro_profile = np.clip(hydro_profile, 0, 100)
    
    # Load Generation
    df = pd.DataFrame({
        'timestamp': dates,
        'Solar': solar_profile,
        'Wind': wind_profile,
        'Nuclear': nuclear_profile,
        'Geothermal': geothermal_profile,
        'Hydro': hydro_profile
    })
    
    total_load = np.zeros(len(dates))
    
    if not building_portfolio:
        # Default fallback
        building_portfolio = [{'type': 'Office', 'annual_mwh': 1000}]
        
    for building in building_portfolio:
        b_type = building.get('type', 'Office')
        target_mwh = building.get('annual_mwh', 1000)
        
        # Generate shape
        raw_profile = generate_load_profile_shape(dates, b_type)
        
        # Scale to target MWh
        current_sum = raw_profile.sum()
        if current_sum > 0:
            scaling_factor = (target_mwh * 1000) / current_sum # Convert MWh to kWh? No, let's stick to MWh.
            # Wait, if target is MWh, and profile is MW (power), then sum(MW * 1h) = MWh.
            # So scaling factor = target_mwh / current_sum
            scaling_factor = target_mwh / current_sum
            final_profile = raw_profile * scaling_factor
        else:
            final_profile = raw_profile
            
        col_name = f"Load_{b_type}_{random.randint(100,999)}" # Unique name in case of duplicates
        # Actually, let's just use type and index if needed, or just append.
        # But user might have multiple "Office" buildings.
        # Let's just call it Load_{Type}. If duplicate, pandas handles it or we should be careful.
        # Simple approach: Load_{Type}
        
        # Check if column exists
        base_name = f"Load_{b_type}"
        count = 1
        while base_name in df.columns:
            count += 1
            base_name = f"Load_{b_type}_{count}"
            
        df[base_name] = final_profile
        total_load += final_profile
        
    df['Load'] = total_load
    
    return df

def calculate_portfolio_metrics(df, solar_capacity, wind_capacity, load_scaling=1.0, region="National Average", base_rec_price=0.50, battery_capacity_mwh=0.0, battery_efficiency=0.85, nuclear_capacity=0.0, geothermal_capacity=0.0, hydro_capacity=0.0):
    """
    Calculates portfolio metrics based on inputs.
    df: DataFrame with 'Solar', 'Wind', 'Nuclear', 'Geothermal', 'Hydro', 'Load' columns
    solar_capacity: MW
    wind_capacity: MW
    nuclear_capacity: MW
    geothermal_capacity: MW
    hydro_capacity: MW
    load_scaling: Multiplier for the base load profile
    """
    # Scale profiles
    if 'Solar' in df.columns and df['Solar'].max() > 0:
        df['Solar_Gen'] = (df['Solar'] / df['Solar'].max()) * solar_capacity
    else:
        df['Solar_Gen'] = 0
        
    if 'Wind' in df.columns and df['Wind'].max() > 0:
        df['Wind_Gen'] = (df['Wind'] / df['Wind'].max()) * wind_capacity
    else:
        df['Wind_Gen'] = 0
    
    if 'Nuclear' in df.columns and df['Nuclear'].max() > 0:
        df['Nuclear_Gen'] = (df['Nuclear'] / df['Nuclear'].max()) * nuclear_capacity
    else:
        df['Nuclear_Gen'] = 0
    
    if 'Geothermal' in df.columns and df['Geothermal'].max() > 0:
        df['Geothermal_Gen'] = (df['Geothermal'] / df['Geothermal'].max()) * geothermal_capacity
    else:
        df['Geothermal_Gen'] = 0
    
    if 'Hydro' in df.columns and df['Hydro'].max() > 0:
        df['Hydro_Gen'] = (df['Hydro'] / df['Hydro'].max()) * hydro_capacity
    else:
        df['Hydro_Gen'] = 0
        
    if 'Load' in df.columns:
        df['Load_Actual'] = df['Load'] * load_scaling
    else:
        df['Load_Actual'] = 0

    # Total Renewable Generation (including all sources)
    df['Total_Renewable_Gen'] = df['Solar_Gen'] + df['Wind_Gen'] + df['Nuclear_Gen'] + df['Geothermal_Gen'] + df['Hydro_Gen']
    
    # Metrics
    total_load = df['Load_Actual'].sum()
    total_gen = df['Total_Renewable_Gen'].sum()
    
    # Annual renewable percent
    annual_re_percent = (total_gen / total_load * 100) if total_load > 0 else 0
    
    # Battery Storage Optimization
    if battery_capacity_mwh > 0:
        # Initialize battery state tracking
        df['Battery_SOC'] = 0.0  # State of Charge (MWh)
        df['Battery_Charge'] = 0.0  # MWh charged this hour
        df['Battery_Discharge'] = 0.0  # MWh discharged this hour
        
        # Simple greedy optimization: charge when excess, discharge when deficit
        soc = 0.0  # Starting state of charge
        
        for i in range(len(df)):
            hourly_surplus = df.loc[i, 'Total_Renewable_Gen'] - df.loc[i, 'Load_Actual']
            
            if hourly_surplus > 0:
                # Excess generation - charge battery
                available_capacity = battery_capacity_mwh - soc
                charge_amount = min(hourly_surplus, available_capacity)
                df.loc[i, 'Battery_Charge'] = charge_amount
                soc += charge_amount * battery_efficiency  # Apply charging efficiency
                
            elif hourly_surplus < 0:
                # Deficit - discharge battery
                deficit = abs(hourly_surplus)
                discharge_amount = min(deficit, soc / battery_efficiency)  # Account for discharge efficiency
                df.loc[i, 'Battery_Discharge'] = discharge_amount
                soc -= discharge_amount * battery_efficiency
            
            df.loc[i, 'Battery_SOC'] = soc
        
        # Update generation to include battery discharge
        df['Total_Renewable_Gen_With_Battery'] = df['Total_Renewable_Gen'] + df['Battery_Discharge'] - df['Battery_Charge']
        # Use battery-adjusted generation for CFE calculation
        df['Hourly_CFE_MWh'] = np.minimum(df['Total_Renewable_Gen_With_Battery'], df['Load_Actual'])
        
        # Define Effective Generation for metrics
        df['Effective_Gen'] = df['Total_Renewable_Gen_With_Battery']
    else:
        # No battery
        df['Hourly_CFE_MWh'] = np.minimum(df['Total_Renewable_Gen'], df['Load_Actual'])
        df['Effective_Gen'] = df['Total_Renewable_Gen']
    
    df['Hourly_CFE_Ratio'] = np.where(df['Load_Actual'] > 0, df['Hourly_CFE_MWh'] / df['Load_Actual'], 1.0)
    
    # Uncapped Ratio for Heatmap Toggle
    df['Hourly_Renewable_Ratio'] = np.where(df['Load_Actual'] > 0, df['Effective_Gen'] / df['Load_Actual'], 0.0)
    
    # Calculate Metrics
    total_annual_load = df['Load_Actual'].sum()
    total_renewable_gen = df['Effective_Gen'].sum()
    
    # MW Match Productivity
    # Sum of min(Gen, Load) / Total Installed MW
    matched_energy_mwh = df[['Effective_Gen', 'Load_Actual']].min(axis=1).sum()
    
    # CFE Score (Volumetric: Total Matched / Total Load)
    # "Add up both and then do the %"
    cfe_score = 0.0
    if total_annual_load > 0:
        cfe_score = (matched_energy_mwh / total_annual_load) * 100
    
    # Loss of Green Hour (Hours where Gen < Load)
    loss_of_green_hours = df[df['Effective_Gen'] < df['Load_Actual']].shape[0]
    loss_of_green_hour_percent = (loss_of_green_hours / 8760) * 100
    
    # Overgeneration (Gen - Load, only positive values)
    overgeneration = (df['Effective_Gen'] - df['Load_Actual']).clip(lower=0).sum()
    
    # Grid Consumption (Load - Gen, only positive values)
    grid_consumption = (df['Load_Actual'] - df['Effective_Gen']).clip(lower=0).sum()
    
    # Emissions
    # Get eGRID factor for the region
    egrid_factor_lb = EGRID_FACTORS.get(region, EGRID_FACTORS["National Average"])
    # Convert to Metric Tons (1 lb = 0.000453592 MT)
    lb_to_mt = 0.000453592
    
    # Grid Emissions: Emissions from grid consumption
    grid_emissions_mt = grid_consumption * egrid_factor_lb * lb_to_mt
    
    # Avoided Emissions: Emissions avoided by renewable generation (assuming it displaces grid power)
    # This is a simplified view; often avoided emissions use marginal rates, but we'll use average for now as per typical simple calculators.
    # We'll calculate it based on the TOTAL renewable generation, as if that MWh replaced grid MWh.
    avoided_emissions_mt = total_renewable_gen * egrid_factor_lb * lb_to_mt
    
    # Location Based Emissions: Total emissions if no renewables were used (Total Load * Grid Factor)
    location_based_emissions_mt = total_annual_load * egrid_factor_lb * lb_to_mt

    # MW Match Productivity
    # Sum of min(Gen, Load) / Total Installed MW
    # matched_energy_mwh is already calculated above
    total_capacity_mw = solar_capacity + wind_capacity
    mw_match_productivity = 0.0
    if total_capacity_mw > 0:
        mw_match_productivity = matched_energy_mwh / total_capacity_mw
    
    # REC Financials
    # Calculate Hourly Net Load (Load - Gen)
    # Positive = Deficit (Need to buy RECs)
    # Negative = Surplus (Can sell RECs)
    # Use battery-adjusted generation if battery is present
    # Calculate Hourly Net Load (Load - Gen)
    # Positive = Deficit (Need to buy RECs)
    # Negative = Surplus (Can sell RECs)
    df['Net_Load_MWh'] = df['Load_Actual'] - df['Effective_Gen']
    
    # Define Pricing Categories
    # We need Month and Hour
    # Assuming df has a datetime index or we can infer from position if it's 8760
    # The generate_synthetic_8760_data function creates a 'timestamp' column.
    
    # Base Price (passed as argument or default)
    # Initialize REC Price column with Base Price
    df['REC_Price_USD'] = base_rec_price
    
    # Extract time components if not already available
    if 'Month' not in df.columns:
        df['Month'] = df['timestamp'].dt.month
    if 'Hour' not in df.columns:
        df['Hour'] = df['timestamp'].dt.hour
        
    # --- Categorization Logic (Fixed Grid-Based Patterns) ---
    
    # Cat 6: Critical Scarcity (Winter Evening Peak: 18:00-20:00 Dec-Feb)
    # These are the hours when grid-wide scarcity is typically highest
    mask_cat6 = (df['Month'].isin([12, 1, 2])) & (df['Hour'].isin([18, 19, 20]))
    df.loc[mask_cat6, 'REC_Price_USD'] = 20.00
    
    # Cat 5: Winter Morning Scarcity (06:00–09:00 Dec–Feb) -> Hours 6, 7, 8
    mask_cat5 = (df['Month'].isin([12, 1, 2])) & (df['Hour'].isin([6, 7, 8])) & (~mask_cat6)
    df.loc[mask_cat5, 'REC_Price_USD'] = 7.00
    
    # Cat 4: Evening Peak (17:00–21:00 Most days) -> Hours 17, 18, 19, 20, 21
    mask_cat4 = (df['Hour'].isin([17, 18, 19, 20, 21])) & (~mask_cat6) & (~mask_cat5)
    df.loc[mask_cat4, 'REC_Price_USD'] = 10.00
    
    # Cat 3: Shoulder Daylight (07:00–10:00 & 15:00–18:00) -> Hours 7, 8, 9, 15, 16, 17
    mask_cat3_hours = df['Hour'].isin([7, 8, 9, 15, 16])
    mask_cat3 = mask_cat3_hours & (~mask_cat6) & (~mask_cat5) & (~mask_cat4)
    df.loc[mask_cat3, 'REC_Price_USD'] = 3.00
    
    # Cat 1: Super-abundant mid-day (10:00–15:00 Mar–Oct) -> Hours 10, 11, 12, 13, 14
    mask_cat1 = (df['Month'].isin(range(3, 11))) & (df['Hour'].isin([10, 11, 12, 13, 14])) & (~mask_cat6)
    df.loc[mask_cat1, 'REC_Price_USD'] = 0.25
    
    # Cat 2: Typical mid-day (10:00–15:00 Nov-Feb) -> Hours 10, 11, 12, 13, 14
    mask_cat2 = (df['Month'].isin([1, 2, 11, 12])) & (df['Hour'].isin([10, 11, 12, 13, 14])) & (~mask_cat6)
    df.loc[mask_cat2, 'REC_Price_USD'] = 1.00
    
    # Calculate Costs and Revenues
    # Cost: When Net Load > 0 (Deficit) -> Negative Value (Outflow)
    df['REC_Cost'] = np.where(df['Net_Load_MWh'] > 0, -df['Net_Load_MWh'] * df['REC_Price_USD'], 0)
    
    # Revenue: When Net Load < 0 (Surplus) -> Overgeneration -> Positive Value (Inflow)
    # We sell the surplus RECs.
    # Note: Overgeneration is defined as Gen - Load (positive). Net Load is Load - Gen (negative).
    # So Surplus = -Net_Load_MWh
    df['REC_Revenue'] = np.where(df['Net_Load_MWh'] < 0, -df['Net_Load_MWh'] * df['REC_Price_USD'], 0)
    
    total_rec_cost = df['REC_Cost'].sum()
    total_rec_revenue = df['REC_Revenue'].sum()
    # Net = Revenue + Cost (where Cost is negative)
    net_rec_cost = total_rec_cost + total_rec_revenue
    
    results = {
        "total_annual_load": total_annual_load,
        "total_renewable_gen": total_renewable_gen,
        "annual_re_percent": (total_renewable_gen / total_annual_load * 100) if total_annual_load > 0 else 0,
        "cfe_percent": cfe_score,
        "loss_of_green_hour_percent": loss_of_green_hour_percent,
        "overgeneration": overgeneration,
        "grid_consumption": grid_consumption,
        "grid_emissions_mt": grid_emissions_mt,
        "avoided_emissions_mt": avoided_emissions_mt,
        "location_based_emissions_mt": location_based_emissions_mt,
        "mw_match_productivity": mw_match_productivity,
        "total_rec_cost": total_rec_cost,
        "total_rec_revenue": total_rec_revenue,
        "net_rec_cost": net_rec_cost,
        "egrid_factor_lb": egrid_factor_lb
    }
    
    # Rename base columns to avoid confusion
    rename_map = {
        'Solar': 'Solar Capacity Factor',
        'Wind': 'Wind Capacity Factor',
        'Load': 'Load Profile'
    }
    df = df.rename(columns=rename_map)
    
    return results, df

def create_zip_export(results, df, portfolio_name, region, inputs=None):
    """
    Creates a zip file containing the JSON summary and the CSV dataset.
    """
    # 1. JSON Summary
    summary_dict = {
        "portfolio_name": portfolio_name,
        "region": region,
        "inputs": inputs if inputs else {},
        "results": results,
        "metadata": {
            "columns": list(df.columns),
            "generated_at": pd.Timestamp.now().isoformat()
        }
    }
    json_str = json.dumps(summary_dict, indent=4)
    
    # 2. CSV Dataset
    # "csv always needs timestamps for each hour and a new column for hourly Carbon free electricity"
    # We already calculated 'Hourly_CFE_Ratio' in the df.
    # Ensure timestamps are present.
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # 3. Zip File
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"{portfolio_name}_summary.json", json_str)
        zip_file.writestr(f"{portfolio_name}_8760_data.csv", csv_buffer.getvalue())
        
    return zip_buffer.getvalue()

def process_uploaded_file(uploaded_file):
    """
    Reads an uploaded CSV or Excel file and standardizes it.
    Expected columns: 'timestamp' (optional), 'Solar', 'Wind', 'Load'.
    If columns are missing, it will try to map common names or fill with zeros/defaults.
    """
    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as z:
                # Find the CSV file
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return None
                # Read the first CSV found
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Standardize column names
        # Simple mapping for demo purposes
        column_map = {
            'Date': 'timestamp', 'Time': 'timestamp', 'datetime': 'timestamp',
            'solar': 'Solar', 'pv': 'Solar', 'Solar Generation': 'Solar',
            'wind': 'Wind', 'Wind Generation': 'Wind',
            'load': 'Load', 'demand': 'Load', 'Consumption': 'Load'
        }
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        if 'Solar' not in df.columns:
            df['Solar'] = 0
        if 'Wind' in df.columns:
            df['Wind'] = 0
        if 'Load' not in df.columns:
            # If no load column, maybe it's just generation data? 
            # For now, let's assume we need load. If missing, maybe use synthetic?
            # Let's just init to 0 and let the user know (in a real app)
            df['Load'] = 0
            
        # Ensure 8760 rows
        if len(df) > 8760:
            df = df.iloc[:8760]
            
        # If timestamp is missing, generate it for a non-leap year (2023)
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='h')
        else:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
        
    except Exception as e:
        return None
