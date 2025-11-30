import pandas as pd

# Test loading the emissions file
print("Testing emissions data loading...")
print()

# Load file
df = pd.read_csv("combinedISOCarbon2024.csv")
print(f"✅ Loaded CSV with {len(df)} rows")
print()

# Test ERCOT filtering
iso_code = "ERCOT"
df['ISO_Code'] = df['ISO_Code'].astype(str).str.strip()
region_df = df[df['ISO_Code'] == iso_code].copy()

print(f"Filtering for '{iso_code}':")
print(f"  Found {len(region_df)} rows")
print()

if not region_df.empty:
    # Parse dates and remove Feb 29
    region_df['period'] = pd.to_datetime(region_df['period'])
    before = len(region_df)
    region_df = region_df[~((region_df['period'].dt.month == 2) & (region_df['period'].dt.day == 29))]
    after = len(region_df)
    print(f"Removed {before - after} leap day rows")
    print(f"  {after} rows remaining")
    print()
    
    # Reindex
    region_df = region_df.sort_values('period')
    emissions_series = region_df.set_index('period')['carbon_intensity_g_kwh']
    
    full_2024_range = pd.date_range(start="2024-01-01", end="2024-12-31 23:00", freq="h")
    full_2024_non_leap = full_2024_range[~((full_2024_range.month == 2) & (full_2024_range.day == 29))]
    
    emissions_series = emissions_series.reindex(full_2024_non_leap)
    
    nan_before = emissions_series.isna().sum()
    emissions_series = emissions_series.ffill().bfill()
    nan_after = emissions_series.isna().sum()
    
    print(f"Filled {nan_before} missing values")
    print(f"  {nan_after} NaN remaining")
    print()
    
    # Convert
    raw_emissions = emissions_series.values * 2.20462
    print(f"Final length: {len(raw_emissions)} (expected 8760)")
    print(f"  First value: {raw_emissions[0]:.1f} lb/MWh")
    print(f"  Last value: {raw_emissions[-1]:.1f} lb/MWh")
    print(f"  Mean: {raw_emissions.mean():.1f} lb/MWh")
    print()
    
    if len(raw_emissions) == 8760:
        print("✅ SUCCESS! Data loaded correctly")
    else:
        print(f"❌ FAILED: Length is {len(raw_emissions)}, not 8760")
else:
    print(f"❌ FAILED: No data found for ISO '{iso_code}'")
