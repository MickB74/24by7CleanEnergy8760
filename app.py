import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random
import time
import os
try:
    import utils
except Exception as e:
    st.error(f"CRITICAL ERROR importing utils: {e}")
    st.stop()

# import importlib
# importlib.reload(utils) # Streamlit handles reloading

# Page Config
st.set_page_config(
    page_title="Eighty760",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize dark mode in session state BEFORE using it
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Inject Custom CSS based on theme
if st.session_state.dark_mode:
    # Dark Mode CSS
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=IBM+Plex+Mono:wght@700&display=swap');

            /* Dark Mode - Override everything */
            .stApp {
                background-color: #0E1117 !important;
            }
            .main .block-container {
                background-color: #0E1117 !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #262730 !important;
            }
            section[data-testid="stSidebar"] > div {
                background-color: #262730 !important;
            }
            
            /* Text Colors */
            .stApp, .stApp p, .stApp label, .stApp span, .stApp div {
                color: #FAFAFA !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #FAFAFA !important;
                font-weight: 700;
            }
            
            /* Global Typography */
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            
            /* Layout Adjustments */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }

            /* Metric Tiles */
            .stMetric {
                background-color: rgba(255, 255, 255, 0.05) !important;
                padding: 1rem;
                border-radius: 8px;
            }
            [data-testid="stMetricValue"] {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 3rem !important;
                font-weight: 700;
                color: #00D9FF !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 1.1rem !important;
                color: #B0B0B0 !important;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            /* Buttons */
            button[kind="primary"], [data-testid="baseButton-primary"] {
                background-color: #00D9FF !important;
                color: #0E1117 !important;
                border: none !important;
                border-radius: 6px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 600 !important;
                width: 100%;
            }
            button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover {
                background-color: #00B8D9 !important;
                color: #0E1117 !important;
            }

            button[kind="secondary"], [data-testid="baseButton-secondary"] {
                background-color: transparent !important;
                color: #FAFAFA !important;
                border: 1px solid #FAFAFA !important;
                border-radius: 6px !important;
            }
            button[kind="secondary"]:hover, [data-testid="baseButton-secondary"]:hover {
                background-color: rgba(255, 255, 255, 0.1) !important;
                color: #FAFAFA !important;
            }

            .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
                border-radius: 6px;
                background-color: #262730 !important;
                color: #FAFAFA !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    # Light Mode CSS
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=IBM+Plex+Mono:wght@700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }
            h1, h2, h3 {
                font-weight: 700;
            }
            
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }

            .stMetric {
                background-color: #F5F5F5;
                padding: 1rem;
                border-radius: 8px;
            }
            [data-testid="stMetricValue"] {
                font-family: 'IBM Plex Mono', monospace;
                font-size: 3rem !important;
                font-weight: 700;
                color: #285477;
            }
            [data-testid="stMetricLabel"] {
                font-size: 1.1rem !important;
                color: #666666;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            button[kind="primary"], [data-testid="baseButton-primary"] {
                background-color: #000000 !important;
                color: #FFFFFF !important;
                border: none !important;
                border-radius: 6px !important;
                padding: 0.75rem 1.5rem !important;
                font-weight: 600 !important;
                width: 100%;
            }
            button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover {
                background-color: #333333 !important;
                color: #FFFFFF !important;
            }

            button[kind="secondary"], [data-testid="baseButton-secondary"] {
                background-color: transparent !important;
                color: #1A1A1A !important;
                border: 1px solid #1A1A1A !important;
                border-radius: 6px !important;
            }
            button[kind="secondary"]:hover, [data-testid="baseButton-secondary"]:hover {
                background-color: #1A1A1A !important;
                color: #FFFFFF !important;
            }

            .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
                border-radius: 6px;
            }

        /* Remove default Streamlit top padding/decoration if possible */
        /* header {visibility: hidden;} */
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Initialize Inputs in Session State if not present
building_types = ["Office", "Data Center", "Retail", "Residential", "Hospital", "Warehouse"]
for b_type in building_types:
    if f"load_{b_type}" not in st.session_state:
        st.session_state[f"load_{b_type}"] = 0
if "solar_capacity" not in st.session_state:
    st.session_state.solar_capacity = 50.0
if "wind_capacity" not in st.session_state:
    st.session_state.wind_capacity = 50.0
if "region_selector" not in st.session_state:
    st.session_state.region_selector = "ERCOT"

# Header
st.image("logo.png", width=600)
st.markdown("<h2 style='color: #285477;'>8760 CE Simulator (v2.0)</h2>", unsafe_allow_html=True)
st.markdown("---")

# Callbacks
def reset_values():
    st.session_state.solar_capacity = 0.0
    st.session_state.wind_capacity = 0.0
    st.session_state.nuclear_capacity = 0.0
    st.session_state.geothermal_capacity = 0.0
    st.session_state.hydro_capacity = 0.0
    for b_type in building_types:
        st.session_state[f"load_{b_type}"] = 0

def randomize_scenario():
    # Randomize Capacities (only Solar and Wind)
    st.session_state.solar_capacity = float(random.randint(10, 500))
    st.session_state.wind_capacity = float(random.randint(10, 500))
    
    # Randomize Loads
    for b_type in building_types:
        st.session_state[f"load_{b_type}"] = random.randint(0, 20) * 25000

# Sidebar Inputs
with st.sidebar:
    st.subheader("Restore Session")
    
    def restore_session_callback():
        uploaded_file = st.session_state.restore_uploader
        if uploaded_file is not None:
            try:
                import zipfile
                import io
                import json
                
                with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
                    # Find JSON and CSV
                    json_files = [f for f in z.namelist() if f.endswith('_summary.json')]
                    csv_files = [f for f in z.namelist() if f.endswith('_8760_data.csv')]
                    
                    if json_files and csv_files:
                        # 1. Restore Inputs from JSON
                        with z.open(json_files[0]) as f:
                            summary_data = json.load(f)
                            inputs = summary_data.get('inputs', {})
                            
                            if inputs:
                                # Update session state with restored inputs
                                st.session_state.solar_capacity = inputs.get('solar_capacity', 0.0)
                                st.session_state.wind_capacity = inputs.get('wind_capacity', 0.0)
                                st.session_state.nuclear_capacity = inputs.get('nuclear_capacity', 0.0)
                                st.session_state.geothermal_capacity = inputs.get('geothermal_capacity', 0.0)
                                st.session_state.hydro_capacity = inputs.get('hydro_capacity', 0.0)
                                st.session_state.battery_capacity = inputs.get('battery_capacity', 0.0)
                                st.session_state.region_selector = inputs.get('region', "ERCOT")
                                
                                # Restore loads
                                for b_type in building_types:
                                    load_key = f"load_{b_type}"
                                    if load_key in inputs:
                                        st.session_state[load_key] = inputs[load_key]
                                
                                st.toast("✓ Settings restored", icon="✅")
                            else:
                                st.toast("⚠ No input settings found in JSON (older export?)", icon="⚠️")

                        # 2. Restore Data from CSV
                        with z.open(csv_files[0]) as f:
                            restored_df = pd.read_csv(f)
                            # Clean columns
                            restored_df.columns = restored_df.columns.str.strip()
                            
                            # Convert timestamp to datetime
                            if 'timestamp' in restored_df.columns:
                                restored_df['timestamp'] = pd.to_datetime(restored_df['timestamp'])
                                
                            # Reconstruct results dict if needed, or use the one from JSON
                            restored_results = summary_data.get('results', {})
                            
                            st.session_state.portfolio_data = {
                                "results": restored_results,
                                "df": restored_df,
                                "region": st.session_state.get('region_selector', "ERCOT"),
                                "solar_capacity": st.session_state.get('solar_capacity', 0.0),
                                "wind_capacity": st.session_state.get('wind_capacity', 0.0),
                                "nuclear_capacity": st.session_state.get('nuclear_capacity', 0.0),
                                "geothermal_capacity": st.session_state.get('geothermal_capacity', 0.0),
                                "hydro_capacity": st.session_state.get('hydro_capacity', 0.0)
                            }
                            st.session_state.analysis_complete = True
                            st.toast("✓ Data restored!", icon="✅")
                    else:
                        st.error("❌ Invalid ZIP format. Must contain _summary.json and _8760_data.csv")
            except Exception as e:
                st.error(f"❌ Error restoring session: {str(e)}")

    st.file_uploader("Upload Exported ZIP", type=['zip'], key="restore_uploader", on_change=restore_session_callback)
            
    st.markdown("---")
    st.subheader("Configuration")

    # Control Buttons with Custom Colors
    st.markdown("""
        <style>
        /* Random and Reset All buttons */
        button[key="random_btn"], button[key="reset_btn"] {
            background-color: #285477 !important;
            color: white !important;
            border: none !important;
        }
        button[key="random_btn"]:hover, button[key="reset_btn"]:hover {
            background-color: #1e3f5a !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    cb1, cb2 = st.columns(2)
    with cb1:
        st.button("Random", on_click=randomize_scenario, use_container_width=True, key="random_btn")
    with cb2:
        st.button("Reset All", on_click=reset_values, use_container_width=True, key="reset_btn")

    st.markdown("<br>", unsafe_allow_html=True)

    # Load Source Selection (outside form for immediate response)
    st.markdown("### 1. Load Profile")
    load_source = st.radio("Load Source", ["Estimate Load", "Upload File"], label_visibility="collapsed", key="load_source_radio")
    
    # File Uploader (outside form for immediate response)
    uploaded_load_file = None
    if load_source == "Upload File":
        uploaded_load_file = st.file_uploader("Upload Load Data", type=['csv', 'xlsx', 'zip'], key="load_uploader")
        if uploaded_load_file is not None:
            try:
                # Handle ZIP files (exported from this app)
                if uploaded_load_file.name.endswith('.zip'):
                    import zipfile
                    import io
                    
                    with zipfile.ZipFile(io.BytesIO(uploaded_load_file.read())) as z:
                        # Look for a CSV file ending with '_8760_data.csv'
                        csv_files = [f for f in z.namelist() if f.endswith('_8760_data.csv')]
                        
                        if csv_files:
                            # Use the first matching CSV file
                            with z.open(csv_files[0]) as f:
                                uploaded_df = pd.read_csv(f)
                            st.success(f"✓ ZIP file uploaded: {uploaded_load_file.name} (found {csv_files[0]})")
                        else:
                            st.error("❌ ZIP file must contain a file ending with '_8760_data.csv'")
                            uploaded_load_file = None
                            uploaded_df = None
                # Handle CSV files
                elif uploaded_load_file.name.endswith('.csv'):
                    uploaded_df = pd.read_csv(uploaded_load_file)
                # Handle Excel files
                else:
                    uploaded_df = pd.read_excel(uploaded_load_file)
                
                if uploaded_df is not None:
                    # Clean column names (strip whitespace)
                    uploaded_df.columns = uploaded_df.columns.str.strip()
                    
                    # Validate the file has required columns
                    valid_load_cols = ['load', 'Load', 'Load_MWh', 'Load_Actual']
                    has_load_col = any(col in uploaded_df.columns for col in valid_load_cols)
                    
                    if has_load_col:
                        if not uploaded_load_file.name.endswith('.zip'):
                            st.success(f"✓ File uploaded: {uploaded_load_file.name} ({len(uploaded_df)} rows)")
                        # Store in session state for later use
                        st.session_state.uploaded_load_data = uploaded_df
                    else:
                        st.error(f"❌ Could not find load column in uploaded file. Found columns: {list(uploaded_df.columns)}")
                        st.info("Expected one of: 'Load', 'load', 'Load_MWh', 'Load_Actual'")
                        uploaded_load_file = None
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                uploaded_load_file = None
        else:
            st.info("📁 Drag & drop or browse to upload a CSV, Excel, or ZIP file with hourly load data")

    with st.form("analysis_config"):
        # Region
        region = st.selectbox("Region", ["ERCOT", "PJM", "CAISO", "MISO", "SPP", "NYISO", "ISO-NE"], key="region_selector")
        
        # Emissions Source
        emissions_source = st.radio("Emissions Data Source", ["Hourly (CSV)", "Annual eGRID"], index=1, horizontal=True)
        emissions_logic = "hourly" if emissions_source == "Hourly (CSV)" else "egrid"

        # Load inputs (only for Estimate Load)
        load_inputs = {}
        if load_source == "Estimate Load":
            # Simplified Load Inputs for UI cleanliness, mapped to existing logic
            st.caption("Define annual consumption for building types.")
            
            for i, b_type in enumerate(building_types):
                val = st.number_input(
                    f"{b_type} (MWh)",
                    min_value=0,
                    step=50000,
                    key=f"load_{b_type}"
                )
                load_inputs[b_type] = val

        st.markdown("###2. Renewables")
        st.caption("Define capacity (MW) for generation assets.")
        solar_capacity = st.number_input("Solar (MW)", min_value=0.0, step=10.0, key="solar_capacity")
        wind_capacity = st.number_input("Wind (MW)", min_value=0.0, step=10.0, key="wind_capacity")
        nuclear_capacity = st.number_input("Nuclear (MW)", min_value=0.0, step=10.0, value=0.0, key="nuclear_capacity")
        geothermal_capacity = st.number_input("Geothermal (MW)", min_value=0.0, step=10.0, value=0.0, key="geothermal_capacity")
        hydro_capacity = st.number_input("Hydropower (MW)", min_value=0.0, step=10.0, value=0.0, key="hydro_capacity")

        st.markdown("### 3. Storage")
        battery_capacity = st.number_input("Battery Capacity (MWh)", min_value=0.0, step=100.0, value=0.0, help="Battery storage capacity for optimized charge/discharge")

        st.markdown("### 4. Financials")
        base_rec_price = st.number_input("Base REC Price ($/MWh)", value=8.00, step=0.50, min_value=0.0, help="Default based on Green-e certified national REC market prices")

        st.markdown("<br>", unsafe_allow_html=True)

        generate_clicked = st.form_submit_button("Generate Analysis", type="primary")

        if generate_clicked:
            with st.spinner("Calculating..."):
                # Check if we're using uploaded data or estimated load
                if load_source == "Upload File" and 'uploaded_load_data' in st.session_state:
                    # Use uploaded load data
                    uploaded_df = st.session_state.uploaded_load_data
                    
                    # Find the load column (case-insensitive)
                    load_col = None
                    for col in uploaded_df.columns:
                        if col.lower() in ['load', 'load_mwh']:
                            load_col = col
                            break
                    
                    if load_col:
                        # Generate base synthetic data structure
                        df = utils.generate_synthetic_8760_data(year=2023, building_portfolio=[], region=region)
                        
                        # Replace the Load column with uploaded data
                        if len(uploaded_df) == 8760:
                            df['Load'] = uploaded_df[load_col].values
                        else:
                            st.warning(f"⚠️ Uploaded file has {len(uploaded_df)} rows, expected 8760. Using first 8760 rows or padding with zeros.")
                            if len(uploaded_df) > 8760:
                                df['Load'] = uploaded_df[load_col].iloc[:8760].values
                            else:
                                # Pad with zeros if less than 8760
                                padded_load = list(uploaded_df[load_col].values) + [0] * (8760 - len(uploaded_df))
                                df['Load'] = padded_load
                    else:
                        st.error("❌ Could not find load column in uploaded file")
                        df = None
                else:
                    # Use estimated load from building types
                    portfolio_list = []
                    for b_type, val in load_inputs.items():
                        if val > 0:
                            portfolio_list.append({'type': b_type, 'annual_mwh': val})
                    
                    # Generate Data
                    df = utils.generate_synthetic_8760_data(year=2023, building_portfolio=portfolio_list, region=region, seed=42)
                
                if df is not None:
                    # Load Hourly Emissions Data if available
                    hourly_emissions = None
                    try:
                        if emissions_logic == "hourly":
                            # Map region to ISO Code
                            region_to_iso = {
                                "ERCOT": "ERCOT",
                                "PJM": "PJM",
                                "MISO": "MISO",
                                "CAISO": "CAISO",
                                "NYISO": "NYISO",
                                "ISO-NE": "ISO-NE",
                                "SPP": "SPP"
                            }
                            iso_code = region_to_iso.get(region, region).strip()
                            
                            st.info(f"🔍 STEP 1: Filtering for ISO code: '{iso_code}'")

                            # Load the combined file using absolute path
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            file_path = os.path.join(current_dir, "combinedISOCarbon2024.csv")
                            
                            combined_em_df = None
                            if os.path.exists(file_path):
                                combined_em_df = pd.read_csv(file_path)
                                st.info(f"✅ STEP 2: Loaded CSV from {file_path}")
                                st.info(f"   Rows: {len(combined_em_df)}")
                                
                                # Show unique ISO codes in the file
                                unique_isos = combined_em_df['ISO_Code'].unique()
                                st.info(f"📋 STEP 3: Found ISOs in file: {', '.join(map(str, unique_isos))}")
                            else:
                                st.error(f"❌ File not found at: {file_path}")
                                raise FileNotFoundError(f"combinedISOCarbon2024.csv not found at {file_path}")

                            # 1. Filter and Clean
                            combined_em_df['ISO_Code'] = combined_em_df['ISO_Code'].astype(str).str.strip()
                            region_emissions_df = combined_em_df[combined_em_df['ISO_Code'] == iso_code].copy()
                            
                            st.info(f"🎯 STEP 4: After filtering for '{iso_code}', found {len(region_emissions_df)} rows")
                            
                            if not region_emissions_df.empty and 'carbon_intensity_g_kwh' in region_emissions_df.columns:
                                st.info(f"✅ STEP 5: Data valid, starting normalization...")
                                
                                # 2. Parse Dates and Remove Leap Day
                                region_emissions_df['period'] = pd.to_datetime(region_emissions_df['period'])
                                
                                # Filter out Feb 29th (Leap Day)
                                before_leap_removal = len(region_emissions_df)
                                region_emissions_df = region_emissions_df[
                                    ~((region_emissions_df['period'].dt.month == 2) & (region_emissions_df['period'].dt.day == 29))
                                ]
                                after_leap_removal = len(region_emissions_df)
                                st.info(f"📅 STEP 6: Removed {before_leap_removal - after_leap_removal} leap day rows. Now {after_leap_removal} rows")
                                
                                # 3. Reindex to Standard 8760 Hours (2023 base year for simulation)
                                # Strategy: Sort by time, reset index, and reindex to 0-8759
                                region_emissions_df = region_emissions_df.sort_values('period')
                                
                                # Extract emissions series
                                emissions_series = region_emissions_df.set_index('period')['carbon_intensity_g_kwh']
                                
                                # Create a target index for 2024 EXCLUDING Feb 29
                                full_2024_range = pd.date_range(start="2024-01-01", end="2024-12-31 23:00", freq="h")
                                full_2024_non_leap = full_2024_range[~((full_2024_range.month == 2) & (full_2024_range.day == 29))]
                                
                                # Reindex source data to this expected range (fills gaps with NaN)
                                emissions_series = emissions_series.reindex(full_2024_non_leap)
                                
                                # 4. Infer Missing Data
                                nan_count_before = emissions_series.isna().sum()
                                emissions_series = emissions_series.ffill().bfill()
                                nan_count_after = emissions_series.isna().sum()
                                st.info(f"🔧 STEP 7: Filled {nan_count_before} missing values. Remaining NaN: {nan_count_after}")
                                
                                # 5. Convert and Finalize
                                raw_emissions = emissions_series.values * 2.20462 # Convert to lb/MWh
                                
                                # Ensure exactly 8760 length
                                st.info(f"📏 STEP 8: Final data length: {len(raw_emissions)} (expected 8760)")
                                
                                if len(raw_emissions) == 8760:
                                    hourly_emissions = pd.Series(raw_emissions) # 0-based index automatically
                                    st.success(f"✅ STEP 9: SUCCESS! Loaded and normalized hourly emissions for {region} (ISO: {iso_code})")
                                    st.info(f"📊 Data Check: First={hourly_emissions.iloc[0]:.1f}, Last={hourly_emissions.iloc[-1]:.1f}, Mean={hourly_emissions.mean():.1f}")
                                else:
                                    # Fallback if something went wrong with length
                                    st.error(f"❌ STEP 9: FAILED - Normalized data length mismatch: {len(raw_emissions)} rows. Expected 8760.")
                                    hourly_emissions = None

                            else:
                                if region_emissions_df.empty:
                                    st.error(f"❌ STEP 5: FAILED - No data found for ISO '{iso_code}' after filtering!")
                                else:
                                    st.error(f"❌ STEP 5: FAILED - Missing 'carbon_intensity_g_kwh' column!")
                                st.warning(f"⚠️ Could not find emissions data for {region} (ISO: {iso_code}). Using eGRID default.")
                                


                    except Exception as e:
                        st.error(f"❌ CRITICAL ERROR while loading emissions data: {str(e)}")
                        st.error(f"Exception type: {type(e).__name__}")
                        import traceback
                        st.code(traceback.format_exc())
                        hourly_emissions = None

                    # CRITICAL VALIDATION: Stop if hourly was requested but not available
                    if emissions_logic == "hourly" and hourly_emissions is None:
                        st.error("🛑 **ANALYSIS STOPPED**")
                        st.error(f"**You selected 'Hourly (CSV)' but hourly emissions data could not be loaded for {region}.**")
                        st.error("**Possible causes:**")
                        st.error("1. The `combinedISOCarbon2024.csv` file is missing or corrupted")
                        st.error(f"2. No data exists for ISO code '{region}' in the file")
                        st.error("3. The file format is incorrect (missing required columns)")
                        st.error("")
                        st.error("**What to do:**")
                        st.error("- Check that `combinedISOCarbon2024.csv` exists in the project directory")
                        st.error("- Verify the file contains data for your selected region")
                        st.error("- OR switch to 'Annual eGRID' emissions source to continue")
                        st.stop()  # Hard stop - don't continue with analysis

                    # Calculate Metrics
                    results, df_result = utils.calculate_portfolio_metrics(df, solar_capacity, wind_capacity, load_scaling=1.0, region=region, base_rec_price=base_rec_price, battery_capacity_mwh=battery_capacity, nuclear_capacity=nuclear_capacity, geothermal_capacity=geothermal_capacity, hydro_capacity=hydro_capacity, hourly_emissions_lb_mwh=hourly_emissions, emissions_logic=emissions_logic)
                    
                    if emissions_logic == "hourly" and results.get('grid_emissions_hourly_mt') is None:
                        st.toast(f"Hourly data unavailable for {region}. Using eGRID.", icon="⚠️")
                    
                    # Debug: Show which emissions logic is being used
                    st.toast(f"Using {emissions_source} emissions data", icon="📊")
                    
                    st.session_state.portfolio_data = {
                        "results": results,
                        "df": df_result,
                        "region": region,
                        "solar_capacity": solar_capacity,
                        "wind_capacity": wind_capacity,
                        "nuclear_capacity": nuclear_capacity,
                        "geothermal_capacity": geothermal_capacity,
                        "hydro_capacity": hydro_capacity,
                        "emissions_logic": emissions_logic,
                        "emissions_source": emissions_source
                    }
                    st.session_state.analysis_complete = True
                    st.rerun()


# Results Section (Main Area)
if st.session_state.analysis_complete and st.session_state.portfolio_data:
    data = st.session_state.portfolio_data
    results = data['results']
    df = data['df']

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric(
            label="CFE %",
            value=f"{results['cfe_percent']:.1f}%",
            help="The percentage of your total electricity consumption that is matched with carbon-free energy generation on an hourly basis."
        )
    
    with m2:
        st.metric(
            label="Loss of Green Hours",
            value=f"{results['loss_of_green_hour_percent']:.1f}%",
            help="The annual percentage of hours where renewable generation is less than load."
        )
    
    with m3:
        st.metric(
            label="Annual Clean Energy / Load",
            value=f"{results['annual_re_percent']:.1f}%",
            help="The ratio of total annual renewable generation to total annual electricity consumption (not matched hourly)."
        )
    
    with m4:
        # Placeholder for "Hours Fully Matched" - let's calculate it roughly
        fully_matched_hours = (df['Hourly_CFE_Ratio'] >= 0.99).sum()
        st.metric(
            label="Hours Matched",
            value=f"{fully_matched_hours:,}",
            help="The number of hours in the year where generation fully meets or exceeds load."
        )

    # Total Generation and Load Row
    m_gen, m_load = st.columns(2)
    with m_gen:
        st.metric(
            label="Total Clean Energy Generation",
            value=f"{results['total_renewable_gen']:,.0f} MWh",
            help="Total renewable energy generated annually."
        )
        # Add breakdown by source with capacity factors
        breakdown_parts = []
        
        # Helper function to calculate capacity factor
        def calc_cf(generation_mwh, capacity_mw):
            if capacity_mw > 0:
                return (generation_mwh / (capacity_mw * 8760)) * 100
            return 0
        
        if df['Solar_Gen'].sum() > 0:
            solar_cf = calc_cf(df['Solar_Gen'].sum(), data['solar_capacity'])
            breakdown_parts.append(f"Solar: {df['Solar_Gen'].sum():,.0f} MWh ({solar_cf:.1f}% CF)")
        if df['Wind_Gen'].sum() > 0:
            wind_cf = calc_cf(df['Wind_Gen'].sum(), data['wind_capacity'])
            breakdown_parts.append(f"Wind: {df['Wind_Gen'].sum():,.0f} MWh ({wind_cf:.1f}% CF)")
        if 'Nuclear_Gen' in df.columns and df['Nuclear_Gen'].sum() > 0:
            nuclear_cf = calc_cf(df['Nuclear_Gen'].sum(), data['nuclear_capacity'])
            breakdown_parts.append(f"Nuclear: {df['Nuclear_Gen'].sum():,.0f} MWh ({nuclear_cf:.1f}% CF)")
        if 'Geothermal_Gen' in df.columns and df['Geothermal_Gen'].sum() > 0:
            geo_cf = calc_cf(df['Geothermal_Gen'].sum(), data['geothermal_capacity'])
            breakdown_parts.append(f"Geothermal: {df['Geothermal_Gen'].sum():,.0f} MWh ({geo_cf:.1f}% CF)")
        if 'Hydro_Gen' in df.columns and df['Hydro_Gen'].sum() > 0:
            hydro_cf = calc_cf(df['Hydro_Gen'].sum(), data['hydro_capacity'])
            breakdown_parts.append(f"Hydro: {df['Hydro_Gen'].sum():,.0f} MWh ({hydro_cf:.1f}% CF)")
        
        if breakdown_parts:
            st.caption(" • ".join(breakdown_parts))
            
    with m_load:
        st.metric(
            label="Total Load",
            value=f"{results['total_annual_load']:,.0f} MWh",
            help="Total annual electricity consumption."
        )
        # Add breakdown by building type with Load Factor
        load_breakdown_parts = []
        for col in df.columns:
            if col.startswith('Load_') and col != 'Load_Actual':
                load_sum = df[col].sum()
                if load_sum > 0:
                    # Extract building type from column name
                    building_type = col.replace('Load_', '').replace('_', ' ')
                    
                    # Calculate Load Factor: Avg Load / Peak Load
                    # Avg Load = Total MWh / 8760
                    # Peak Load = Max MW in the column
                    peak_load = df[col].max()
                    if peak_load > 0:
                        load_factor = (load_sum / (peak_load * 8760)) * 100
                        load_breakdown_parts.append(f"{building_type}: {load_sum:,.0f} MWh ({load_factor:.1f}% LF)")
                    else:
                        load_breakdown_parts.append(f"{building_type}: {load_sum:,.0f} MWh")
        
        if load_breakdown_parts:
            st.caption(" • ".join(load_breakdown_parts))
            
    # Second Metrics Row
    m5, m6 = st.columns(2)
    
    with m5:
        st.metric(
            label="MWh Needed for 24/7",
            value=f"{results['grid_consumption']:,.0f}",
            help="Total electricity load that could not be met by renewable generation (Grid Consumption)."
        )
            
    with m6:
        st.metric(
            label="MWh Overgenerated for 24/7",
            value=f"{results['overgeneration']:,.0f}",
            help="Total renewable generation in excess of load during hours where generation exceeds load."
        )
            
    st.markdown("---")
    
    # Financials Row
    st.markdown("### REC Financial Analysis")
    
    def format_currency(value):
            if abs(value) >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            elif abs(value) >= 1_000:
                return f"${value/1_000:.1f}k"
            else:
                return f"${value:,.0f}"
    
    f1, f2, f3 = st.columns(3)
    with f1:
        avg_cost_per_mwh = abs(results['total_rec_cost'] / results['grid_consumption']) if results['grid_consumption'] > 0 else 0
        st.metric(
            label="Total REC Cost for 24/7",
            value=format_currency(results['total_rec_cost']),
            help="Cost to buy RECs for deficit hours."
        )
        st.caption(f"Avg: ${avg_cost_per_mwh:.2f}/MWh")
    with f2:
        avg_revenue_per_mwh = results['total_rec_revenue'] / results['overgeneration'] if results['overgeneration'] > 0 else 0
        st.metric(
            label="Total REC Revenue from Overgeneration",
            value=format_currency(results['total_rec_revenue']),
            help="Revenue from selling RECs during surplus hours."
        )
        st.caption(f"Avg: ${avg_revenue_per_mwh:.2f}/MWh")
    with f3:
        st.metric(
            label="Net Profit/(Loss)",
            value=format_currency(results['net_rec_cost']),
            help="Total Revenue + Total Cost. Positive means net profit, negative means net loss."
        )
            
    st.markdown("---")
    
    # Carbon Impact Row
    st.markdown("### Carbon Impact")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label="Location Based Emissions",
            value=f"{results['location_based_emissions_mt']:,.0f} MT",
            help="Total emissions if no renewables were used (Total Load * Grid Factor). Note: If there is overgeneration, Avoided + Grid Emissions will be greater than Location Based, as you avoid extra emissions by exporting clean energy."
        )
        st.caption(f"{results['total_annual_load']:,.0f} MWh * {results['egrid_factor_lb']:.1f} lb/MWh")
    with c2:
        st.metric(
            label="Market Based 24/7 Emissions",
            value=f"{results['grid_emissions_mt']:,.0f} MT",
            help="Estimated CO2e emissions from grid electricity consumption that is not hourly matched to clean energy."
        )
        if results.get('grid_emissions_hourly_mt') is not None:
             st.caption(f"{results['grid_consumption']:,.0f} MWh (Grid) * Hourly Factors")
        else:
             st.caption(f"{results['grid_consumption']:,.0f} MWh (Grid) * {results['egrid_factor_lb']:.1f} lb/MWh")
    with c3:
        st.metric(
            label="Consequential Emission Reduction",
            value=f"{results['avoided_emissions_mt']:,.0f} MT",
            help="Carbon factor where the clean energy is generated * Clean Energy Generation."
        )
        if results.get('avoided_emissions_hourly_mt') is not None:
             st.caption(f"{results['effective_gen']:,.0f} MWh (Gen) * Hourly Factors")
        else:
             st.caption(f"{results['effective_gen']:,.0f} MWh (Gen) * {results['egrid_factor_lb']:.1f} lb/MWh")
            
    with st.expander("Show Emissions Calculation Examples"):
        st.caption("Examples of how emissions are calculated for a single hour (randomly selected).")
        if df is not None and not df.empty:
            ex_row = df.sample(1).iloc[0]
            
            # Location Based
            loc_em = ex_row['Load_Actual'] * results['egrid_factor_lb']
            
            # Market Based
            if emissions_logic == 'hourly' and 'Hourly_Grid_Emissions_lb' in ex_row:
                mkt_factor = ex_row['Emissions_Factor_Hourly_lb_MWh'] if 'Emissions_Factor_Hourly_lb_MWh' in ex_row else 0
                mkt_em = ex_row['Hourly_Grid_Emissions_lb']
                mkt_desc = f"{ex_row['Grid_Consumption']:.1f} MWh (Grid) × {mkt_factor:.1f} lb/MWh"
            else:
                mkt_em = ex_row['Grid_Consumption'] * results['egrid_factor_lb']
                mkt_desc = f"{ex_row['Grid_Consumption']:.1f} MWh (Grid) × {results['egrid_factor_lb']:.1f} lb/MWh"
                
            # Consequential
            if emissions_logic == 'hourly' and 'Hourly_Avoided_Emissions_lb' in ex_row:
                avoid_factor = ex_row['Emissions_Factor_Hourly_lb_MWh'] if 'Emissions_Factor_Hourly_lb_MWh' in ex_row else 0
                avoid_em = ex_row['Hourly_Avoided_Emissions_lb']
                avoid_desc = f"{ex_row['Effective_Gen']:.1f} MWh (Gen) × {avoid_factor:.1f} lb/MWh"
            else:
                avoid_em = ex_row['Effective_Gen'] * results['egrid_factor_lb']
                avoid_desc = f"{ex_row['Effective_Gen']:.1f} MWh (Gen) × {results['egrid_factor_lb']:.1f} lb/MWh"

            st.markdown(f"""
            **Time**: {ex_row['timestamp'].strftime('%B %d, %H:00')}
            
            **1. Location Based Emissions**
            - {ex_row['Load_Actual']:.1f} MWh (Total Load) × {results['egrid_factor_lb']:.1f} lb/MWh = **{loc_em:,.1f} lb**
            
            **2. Market Based 24/7 Emissions**
            - {mkt_desc} = **{mkt_em:,.1f} lb**
            
            **3. Consequential Emission Reduction**
            - {avoid_desc} = **{avoid_em:,.1f} lb**
            """)

    st.markdown("---")
    
    # Chart Section
    st.subheader("Load vs Renewables (8760)")
    
    # Simplified Sparkline/Line Chart
    base = alt.Chart(df.reset_index()).encode(x=alt.X('timestamp', title=None, axis=None))
    
    # Define colors
    domain = ['Load', 'Generation']
    range_ = ['#FF00FF', '#000000']

    line_load = base.mark_line(strokeWidth=1).transform_calculate(
            Source="'Load'"
    ).encode(
            y=alt.Y('Load_Actual', title=None, axis=None),
            color=alt.Color('Source:N', scale=alt.Scale(domain=domain, range=range_), title=None)
    )
    
    line_gen = base.mark_line(strokeWidth=1).transform_calculate(
            Source="'Generation'"
    ).encode(
            y=alt.Y('Total_Renewable_Gen', title=None, axis=None),
            color=alt.Color('Source:N', scale=alt.Scale(domain=domain, range=range_), title=None)
    )
    
    chart = alt.layer(line_gen, line_load).properties(
            height=200,
            width='container'
    ).configure_view(strokeWidth=0).configure_legend(
            orient='bottom',
            title=None
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Mini Heatmap (Month x Hour)
    st.subheader("Hourly Match Heatmap")
    
    heatmap_mode = st.radio("Heatmap Mode", ["Capped at 100%", "Total Renewable Generation"], horizontal=True, label_visibility="collapsed")
    
    # Aggregate data for 12x24 grid
    heatmap_data = df.copy()
    heatmap_data['Month'] = heatmap_data['timestamp'].dt.month_name()
    heatmap_data['MonthNum'] = heatmap_data['timestamp'].dt.month
    heatmap_data['Hour'] = heatmap_data['timestamp'].dt.hour
    
    # Determine column to use
    metric_col = 'Hourly_CFE_Ratio' if heatmap_mode == "Capped at 100%" else 'Hourly_Renewable_Ratio'
    
    # Group by Month and Hour
    heatmap_agg = heatmap_data.groupby(['MonthNum', 'Month', 'Hour'])[metric_col].mean().reset_index()
    
    # Sort months
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    
    heatmap = alt.Chart(heatmap_agg).mark_rect().encode(
            x=alt.X('Hour:O', title='Hour of Day'),
            y=alt.Y('Month:N', sort=months, title=None),
            color=alt.Color(metric_col, scale=alt.Scale(scheme='greys', domain=[0, 1] if heatmap_mode == "Capped at 100%" else [0, 2]), title='CFE %' if heatmap_mode == "Capped at 100%" else 'Gen %'),
            tooltip=[
                alt.Tooltip('Month', title='Month'),
                alt.Tooltip('Hour', title='Hour'),
                alt.Tooltip(metric_col, title='Avg Match %', format='.1%')
            ]
    ).properties(
            height=300
    )
    st.altair_chart(heatmap, use_container_width=True)
    
    # REC Price Heatmap
    st.subheader("REC Price Heatmap")
    
    # Aggregate REC prices for 12x24 grid
    rec_heatmap_data = df.copy()
    rec_heatmap_data['Month'] = rec_heatmap_data['timestamp'].dt.month_name()
    rec_heatmap_data['MonthNum'] = rec_heatmap_data['timestamp'].dt.month
    rec_heatmap_data['Hour'] = rec_heatmap_data['timestamp'].dt.hour
    
    # Group by Month and Hour to get average REC price
    rec_heatmap_agg = rec_heatmap_data.groupby(['MonthNum', 'Month', 'Hour'])['REC_Price_USD'].mean().reset_index()
    
    # Create heatmap with color scale for REC prices
    rec_heatmap = alt.Chart(rec_heatmap_agg).mark_rect().encode(
            x=alt.X('Hour:O', title='Hour of Day'),
            y=alt.Y('Month:N', sort=months, title=None),
            color=alt.Color('REC_Price_USD', scale=alt.Scale(scheme='oranges', domain=[0, 20]), title='$/MWh'),
            tooltip=[
                alt.Tooltip('Month', title='Month'),
                alt.Tooltip('Hour', title='Hour'),
                alt.Tooltip('REC_Price_USD', title='Avg REC Price', format='$.2f')
            ]
    ).properties(
            height=300
    )
    st.altair_chart(rec_heatmap, use_container_width=True)
    
    # Grid Emissions Intensity Heatmap
    st.subheader("Grid Emissions Intensity Heatmap (lb/MWh)")
    st.caption("Hourly grid emissions intensity. Darker red indicates higher emissions (dirtier grid).")
    
    # Use the same heatmap_data which has 'Emissions_Factor_lb_MWh'
    # Aggregate by Month and Hour to smooth out variability if needed, or just plot raw if 8760
    # For heatmap, usually we want average per month-hour
    emissions_heatmap_agg = heatmap_data.groupby(['Month', 'Hour'])['Emissions_Factor_lb_MWh'].mean().reset_index()
    
    heatmap_emissions = alt.Chart(emissions_heatmap_agg).mark_rect().encode(
        x=alt.X('Hour:O', title='Hour of Day'),
        y=alt.Y('Month:O', title='Month', sort=months),
        color=alt.Color('Emissions_Factor_lb_MWh:Q', title='lb/MWh', scale=alt.Scale(scheme='reds')),
        tooltip=['Month', 'Hour', alt.Tooltip('Emissions_Factor_lb_MWh:Q', format='.2f', title='Emissions (lb/MWh)')]
    ).properties(
        height=300
    )
    st.altair_chart(heatmap_emissions, use_container_width=True)

    # Net REC Financial Position Heatmap
    st.subheader("Net REC Financial Position Heatmap")
    st.caption("Financial flow from selling excess RECs (Revenue) and buying needed RECs (Cost). Green = Net Revenue, Red = Net Cost.")
    
    # Calculate Net Flow
    fin_heatmap_data = df.copy()
    fin_heatmap_data['Net_REC_Flow'] = fin_heatmap_data['REC_Cost'] + fin_heatmap_data['REC_Revenue']
    fin_heatmap_data['Month'] = fin_heatmap_data['timestamp'].dt.month_name()
    fin_heatmap_data['MonthNum'] = fin_heatmap_data['timestamp'].dt.month
    fin_heatmap_data['Hour'] = fin_heatmap_data['timestamp'].dt.hour
    
    # Group by Month and Hour
    fin_heatmap_agg = fin_heatmap_data.groupby(['MonthNum', 'Month', 'Hour'])['Net_REC_Flow'].mean().reset_index()
    
    # Create heatmap
    # We need a diverging color scale.
    # Find max abs value to center the domain
    max_val = fin_heatmap_agg['Net_REC_Flow'].abs().max()
    
    fin_heatmap = alt.Chart(fin_heatmap_agg).mark_rect().encode(
            x=alt.X('Hour:O', title='Hour of Day'),
            y=alt.Y('Month:N', sort=months, title=None),
            color=alt.Color('Net_REC_Flow', scale=alt.Scale(scheme='redyellowgreen', domain=[-max_val, max_val]), title='Avg Net $'),
            tooltip=[
                alt.Tooltip('Month', title='Month'),
                alt.Tooltip('Hour', title='Hour'),
                alt.Tooltip('Net_REC_Flow', title='Avg Net Flow', format='$.2f')
            ]
    ).properties(
            height=300
    )
    st.altair_chart(fin_heatmap, use_container_width=True)

    
    with st.expander("Show Example Calculations from Data"):
        st.caption("Two actual hours from your simulation (randomly selected)")
        
        # Find a Net Cost Example (Deficit)
        cost_examples = df[df['Net_Load_MWh'] > 0]
        if not cost_examples.empty:
            ex_cost = cost_examples.sample(1).iloc[0]
            cost_calc = ex_cost['Net_Load_MWh'] * ex_cost['REC_Price_USD']
            st.markdown(f"""
            **Example 1: Buying RECs (Net Cost)**
            - **Time**: {ex_cost['timestamp'].strftime('%B %d, %H:00')}
            - **Load**: {ex_cost['Load_Actual']:.1f} MWh
            - **Generation**: {ex_cost['Total_Renewable_Gen']:.1f} MWh
            - **Shortfall**: {ex_cost['Net_Load_MWh']:.1f} MWh
            - **REC Price**: \\${ex_cost['REC_Price_USD']:.2f}/MWh
            - **Calculation**: {ex_cost['Net_Load_MWh']:.1f} MWh × \\${ex_cost['REC_Price_USD']:.2f} = **-\\${abs(ex_cost['REC_Cost']):.2f}**
            """)
        else:
            st.info("No hours with Net Cost found (100% coverage).")
            
        st.markdown("---")
        
        # Find a Net Revenue Example (Surplus)
        rev_examples = df[df['Net_Load_MWh'] < 0]
        if not rev_examples.empty:
            ex_rev = rev_examples.sample(1).iloc[0]
            rev_calc = abs(ex_rev['Net_Load_MWh']) * ex_rev['REC_Price_USD']
            st.markdown(f"""
            **Example 2: Selling RECs (Net Revenue)**
            - **Time**: {ex_rev['timestamp'].strftime('%B %d, %H:00')}
            - **Load**: {ex_rev['Load_Actual']:.1f} MWh
            - **Generation**: {ex_rev['Total_Renewable_Gen']:.1f} MWh
            - **Excess**: {abs(ex_rev['Net_Load_MWh']):.1f} MWh
            - **REC Price**: \\${ex_rev['REC_Price_USD']:.2f}/MWh
            - **Calculation**: {abs(ex_rev['Net_Load_MWh']):.1f} MWh × \\${ex_rev['REC_Price_USD']:.2f} = **+\\${ex_rev['REC_Revenue']:.2f}**
            """)
        else:
            st.info("No hours with Net Revenue found (No overgeneration).")

    # Auto-Insights
    # Auto-Insights Logic
    insights = []
    
    # 1. Generation Dominance
    total_solar = df['Solar_Gen'].sum()
    total_wind = df['Wind_Gen'].sum()
    total_nuclear = df['Nuclear_Gen'].sum() if 'Nuclear_Gen' in df.columns else 0
    total_geothermal = df['Geothermal_Gen'].sum() if 'Geothermal_Gen' in df.columns else 0
    total_hydro = df['Hydro_Gen'].sum() if 'Hydro_Gen' in df.columns else 0
    total_gen = total_solar + total_wind + total_nuclear + total_geothermal + total_hydro
    
    if total_gen > 0:
        # Find dominant source
        sources = {
            'Solar': total_solar,
            'Wind': total_wind,
            'Nuclear': total_nuclear,
            'Geothermal': total_geothermal,
            'Hydropower': total_hydro
        }
        dominant_source = max(sources, key=sources.get)
        dominant_share = sources[dominant_source] / total_gen
        
        if dominant_share > 0.5:
            insights.append(f"**{dominant_source}-dominant profile:** {dominant_share:.0%} of generation comes from {dominant_source} assets.")
        else:
            # Show breakdown of all non-zero sources
            active_sources = {k: v for k, v in sources.items() if v > 0}
            breakdown = ", ".join([f"{k} ({v/total_gen:.0%})" for k, v in active_sources.items()])
            insights.append(f"**Diversified profile:** {breakdown}")
            
    # 2. Deficit Hours (Time of day with lowest CFE)
    # heatmap_agg is already calculated: ['MonthNum', 'Month', 'Hour', 'Hourly_CFE_Ratio']
    # Group by Hour to find worst time of day on average
    hourly_avg_cfe = df.groupby(df['timestamp'].dt.hour)['Hourly_CFE_Ratio'].mean()
    worst_hour = hourly_avg_cfe.idxmin()
    worst_hour_val = hourly_avg_cfe.min()
    
    # Format hour nicely
    worst_hour_str = pd.to_datetime(f"2023-01-01 {worst_hour}:00").strftime("%I %p").lstrip("0")
    insights.append(f"**Lowest match time:** Average CFE drops to {worst_hour_val:.0%} around {worst_hour_str}.")

    # 3. Seasonal Lows
    monthly_avg_gen = df.groupby(df['timestamp'].dt.month_name())['Total_Renewable_Gen'].mean()
    worst_month = monthly_avg_gen.idxmin()
    insights.append(f"**Seasonal low:** Lowest average renewable generation observed in {worst_month}.")

    st.subheader("Auto-Insights")
    st.caption(f"Debug: Solar Cap={st.session_state.get('solar_capacity')}, Wind Cap={st.session_state.get('wind_capacity')}")
    for insight in insights:
        st.markdown(f"* {insight}")
    
    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Collect inputs for export
    export_inputs = {
        "solar_capacity": st.session_state.get('solar_capacity', 0.0),
        "wind_capacity": st.session_state.get('wind_capacity', 0.0),
        "nuclear_capacity": st.session_state.get('nuclear_capacity', 0.0),
        "geothermal_capacity": st.session_state.get('geothermal_capacity', 0.0),
        "hydro_capacity": st.session_state.get('hydro_capacity', 0.0),
        "battery_capacity": st.session_state.get('battery_capacity', 0.0),
        "region": region
    }
    # Add building loads
    for b_type in building_types:
        export_inputs[f"load_{b_type}"] = st.session_state.get(f"load_{b_type}", 0.0)
        
    zip_data = utils.create_zip_export(results, df, "Eighty760_Analysis", region, inputs=export_inputs)
    st.download_button(
            label="Download ZIP", # Label per spec, but functionality is ZIP for now
            data=zip_data,
            file_name="Eighty760_Report.zip",
            mime="application/zip",
            type="secondary"
    )


else:
    # Empty State / Placeholder
    st.info("Configure your portfolio on the left and click 'Generate Analysis' to see results.")
    
    # Visual Placeholder
    st.markdown("""
    <div style="text-align: center; padding: 40px; color: #ccc;">
        <h3>Waiting for Input...</h3>
    </div>
    """, unsafe_allow_html=True)
