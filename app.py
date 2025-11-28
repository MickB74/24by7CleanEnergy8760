import streamlit as st
import pandas as pd
import utils
import time
import random
import altair as alt

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
st.markdown("<h2 style='color: #285477;'>8760 CE Simulator</h2>", unsafe_allow_html=True)
st.markdown("---")

# Callbacks
def reset_values():
    st.session_state.solar_capacity = 0.0
    st.session_state.wind_capacity = 0.0
    for b_type in building_types:
        st.session_state[f"load_{b_type}"] = 0

def randomize_scenario():
    # Randomize Capacities
    st.session_state.solar_capacity = float(random.randint(10, 500))
    st.session_state.wind_capacity = float(random.randint(10, 500))
    
    # Randomize Loads
    for b_type in building_types:
        st.session_state[f"load_{b_type}"] = random.randint(0, 20) * 25000

# Sidebar Inputs
with st.sidebar:
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

    with st.form("analysis_config"):
        # Region
        region = st.selectbox("Region", ["ERCOT", "PJM", "CAISO", "MISO", "SPP", "NYISO", "ISO-NE"], key="region_selector")

        st.markdown("### 1. Load Profile")
        load_source = st.radio("Load Source", ["Estimate Load", "Upload File"], label_visibility="collapsed")

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
        else:
            uploaded_load_file = st.file_uploader("Upload Load CSV", type=['csv', 'xlsx'])
            # Placeholder for file processing logic integration

        st.markdown("### 2. Renewables")
        st.caption("Define capacity (MW) for generation assets.")
        solar_capacity = st.number_input("Solar (MW)", min_value=0.0, step=10.0, key="solar_capacity")
        wind_capacity = st.number_input("Wind (MW)", min_value=0.0, step=10.0, key="wind_capacity")

        st.markdown("### 3. Storage")
        battery_capacity = st.number_input("Battery Capacity (MWh)", min_value=0.0, step=100.0, value=0.0, help="Battery storage capacity for optimized charge/discharge")

        st.markdown("### 4. Financials")
        base_rec_price = st.number_input("Base REC Price ($/MWh)", value=8.00, step=0.50, min_value=0.0, help="Default based on Green-e certified national REC market prices")

        st.markdown("<br>", unsafe_allow_html=True)

        generate_clicked = st.form_submit_button("Generate Analysis", type="primary")

        if generate_clicked:
            with st.spinner("Calculating..."):
                # Logic adapted from previous app.py
                portfolio_list = []
                for b_type, val in load_inputs.items():
                    if val > 0:
                        portfolio_list.append({'type': b_type, 'annual_mwh': val})
                
                # Default if empty
                if not portfolio_list and load_source == "Estimate Load":
                    # Add a dummy load if nothing selected to avoid crash, or handle gracefully
                    pass 

                # Generate Data
                df = utils.generate_synthetic_8760_data(year=2023, building_portfolio=portfolio_list, region=region)
                
                # Calculate Metrics
                results, df_result = utils.calculate_portfolio_metrics(df, solar_capacity, wind_capacity, load_scaling=1.0, region=region, base_rec_price=base_rec_price, battery_capacity_mwh=battery_capacity)
                
                st.session_state.portfolio_data = {
                    "results": results,
                    "df": df_result,
                    "region": region,
                    "solar_capacity": solar_capacity,
                    "wind_capacity": wind_capacity
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
    with m_load:
        st.metric(
            label="Total Load",
            value=f"{results['total_annual_load']:,.0f} MWh",
            help="Total annual electricity consumption."
        )
            
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
            label="Net REC Cost",
            value=format_currency(results['net_rec_cost']),
            help="Total Cost - Total Revenue. Positive means net cost, negative means net profit."
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
        st.caption(f"Grid Factor: {results['egrid_factor_lb']:.1f} lb/MWh")
    with c2:
        st.metric(
            label="Market Based 24/7 Emissions",
            value=f"{results['grid_emissions_mt']:,.0f} MT",
            help="Estimated CO2e emissions from grid electricity consumption that is not hourly matched to clean energy."
        )
    with c3:
        st.metric(
            label="Consequential Emission Reduction",
            value=f"{results['avoided_emissions_mt']:,.0f} MT",
            help="Carbon factor where the clean energy is generated * Clean Energy Generation."
        )
            
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
    total_gen = total_solar + total_wind
    
    if total_gen > 0:
        solar_share = total_solar / total_gen
        if solar_share > 0.6:
            insights.append(f"**Solar-dominant profile:** {solar_share:.0%} of generation comes from solar assets.")
        elif solar_share < 0.4:
            insights.append(f"**Wind-dominant profile:** {(1-solar_share):.0%} of generation comes from wind assets.")
        else:
            insights.append(f"**Balanced profile:** Mix of Solar ({solar_share:.0%}) and Wind ({(1-solar_share):.0%}).")
            
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
    zip_data = utils.create_zip_export(results, df, "Eighty760_Analysis", region)
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
