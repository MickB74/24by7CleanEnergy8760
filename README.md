# 24/7 Clean Energy Analysis Tool

A Streamlit application for analyzing 24/7 Carbon-Free Energy (CFE) portfolios. This tool allows users to simulate energy generation from various renewable sources (Solar, Wind, Nuclear, Geothermal, Hydropower) and battery storage against hourly load profiles to calculate CFE scores, emissions impacts, and financial metrics.

## Features

-   **Multi-Region Support**: Analyze portfolios in ERCOT, PJM, MISO, CAISO, NYISO, ISO-NE, and SPP.
-   **Hourly Emissions Data**: Integrates with 2024 ISO-specific carbon intensity data for accurate emissions modeling.
-   **Comprehensive Generation Modeling**:
    -   **Solar & Wind**: Uses NREL-based synthetic profiles or actual historical data.
    -   **Baseload**: Nuclear and Geothermal modeling.
    -   **Dispatchable**: Hydropower and Battery Storage (with charge/discharge logic).
-   **Financial Analysis**: Calculates Net Present Value (NPV), Internal Rate of Return (IRR), and Payback Period based on energy savings and REC revenue.
-   **Interactive Visualizations**:
    -   Hourly generation vs. load charts.
    -   Grid emissions intensity heatmaps.
    -   Monthly energy mix breakdowns.
    -   CFE score heatmaps.

## Getting Started

### Prerequisites

-   Python 3.8+
-   Streamlit
-   Pandas
-   Plotly
-   Numpy

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/24by7CleanEnergy8760.git
    cd 24by7CleanEnergy8760
    ```

2.  Install dependencies:
    ```bash
    pip install streamlit pandas plotly numpy
    ```

3.  Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

1.  **Load Profile**: Upload your own hourly load data (CSV) or use the built-in estimator based on building type and annual consumption.
2.  **Configuration**: Select your region and emissions data source (Hourly CSV or Annual eGRID).
3.  **Portfolio Design**: Set capacities for Solar, Wind, Nuclear, Geothermal, Hydro, and Battery Storage.
4.  **Generate Analysis**: Click the button to run the simulation.
5.  **Explore Results**: View the dashboard for CFE scores, emissions reductions, and financial performance.

## Data Sources

-   **Emissions**: Hourly grid carbon intensity data from `combinedISOCarbon2024.csv` (sourced from ISO data).
-   **Generation Profiles**: Synthetic profiles derived from NREL PVWatts and other renewable energy models.

## License

[MIT License](LICENSE)
