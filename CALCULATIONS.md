# Definitions and Formulas

This document outlines the definitions and formulas used in the Eighty760 Clean Energy Simulator.

## 1. Generation & Load Scaling

**Solar/Wind Generation (MWh)**
Scales the normalized profile (0-100) to the user-defined capacity.
$$ \text{Gen}_{source} = \left( \frac{\text{Profile}_{source}}{\text{max}(\text{Profile}_{source})} \right) \times \text{Capacity}_{MW} $$

**Actual Load (MWh)**
Scales the base load profile by a scaling factor (if applicable).
$$ \text{Load}_{Actual} = \text{Load}_{Base} \times \text{ScalingFactor} $$

**Total Renewable Generation**
Sum of all renewable sources.
$$ \text{Gen}_{Total} = \text{Gen}_{Solar} + \text{Gen}_{Wind} + \text{Gen}_{Nuclear} + \text{Gen}_{Geothermal} + \text{Gen}_{Hydro} $$

## 2. Battery Storage Logic

The battery uses a greedy optimization strategy: charge when there is excess generation, discharge when there is a deficit.

**Hourly Surplus/Deficit**
$$ \Delta_{Hour} = \text{Gen}_{Total} - \text{Load}_{Actual} $$

**Charging (when $\Delta_{Hour} > 0$)**
$$ \text{Charge} = \min(\Delta_{Hour}, \text{Capacity}_{Battery} - \text{SOC}_{prev}) $$
$$ \text{SOC}_{new} = \text{SOC}_{prev} + (\text{Charge} \times \eta_{efficiency}) $$

**Discharging (when $\Delta_{Hour} < 0$)**
$$ \text{Deficit} = |\Delta_{Hour}| $$
$$ \text{Discharge} = \min(\text{Deficit}, \frac{\text{SOC}_{prev}}{\eta_{efficiency}}) $$
$$ \text{SOC}_{new} = \text{SOC}_{prev} - (\text{Discharge} \times \eta_{efficiency}) $$
*Note: Default efficiency ($\eta$) is 0.85.*

**Effective Generation**
Net generation available to load after battery activity.
$$ \text{Gen}_{Effective} = \text{Gen}_{Total} + \text{Discharge} - \text{Charge} $$

## 3. Carbon Free Energy (CFE) Metrics

**Hourly CFE (MWh)**
The amount of load matched by carbon-free energy in a given hour.
$$ \text{CFE}_{Hour} = \min(\text{Gen}_{Effective}, \text{Load}_{Actual}) $$

**Hourly CFE Ratio**
$$ \text{Ratio}_{CFE} = \frac{\text{CFE}_{Hour}}{\text{Load}_{Actual}} $$
*(Capped at 1.0)*

**Annual CFE Score (%)**
Volumetric score representing the percentage of total load matched by CFE.
$$ \text{Score}_{CFE} = \left( \frac{\sum \text{CFE}_{Hour}}{\sum \text{Load}_{Actual}} \right) \times 100 $$

**MW Match Productivity**
Efficiency of installed capacity in matching load.
$$ \text{Productivity} = \frac{\sum \min(\text{Gen}_{Effective}, \text{Load}_{Actual})}{\text{Capacity}_{Total}} $$

**Loss of Green Hours (%)**
Percentage of hours where generation was insufficient to meet load.
$$ \text{LoGH}_{\%} = \left( \frac{\text{Count}(\text{Gen}_{Effective} < \text{Load}_{Actual})}{8760} \right) \times 100 $$

## 4. Grid Interaction

**Overgeneration (MWh)**
Excess energy sent to the grid (or curtailed) after battery charging.
$$ \text{Overgen} = \max(0, \text{Gen}_{Effective} - \text{Load}_{Actual}) $$

**Grid Consumption (MWh)**
Energy drawn from the grid to meet load deficit.
$$ \text{GridCons} = \max(0, \text{Load}_{Actual} - \text{Gen}_{Effective}) $$

## 5. Emissions

**Grid Emissions (lb)**
Emissions associated with grid consumption.
$$ \text{Emissions}_{Grid} = \text{GridCons} \times \text{Factor}_{Emissions} $$

**Avoided Emissions (lb)**
Emissions avoided by using renewable generation instead of grid power.
$$ \text{Emissions}_{Avoided} = \text{Gen}_{Effective} \times \text{Factor}_{Emissions} $$

**Location-Based Emissions (lb)**
Total emissions if the entire load was served by the grid.
$$ \text{Emissions}_{Location} = \text{Load}_{Actual} \times \text{Factor}_{Emissions} $$

**Implied Annual Emissions Factor (lb/MWh)**
Weighted average emissions factor of the grid electricity actually consumed.
$$ \text{Factor}_{Implied} = \frac{\sum \text{Emissions}_{Grid}}{\sum \text{GridCons}} $$

*Note: $\text{Factor}_{Emissions}$ can be either a static eGRID value or an hourly marginal emissions rate depending on configuration.*

## 6. Financials (REC Pricing)

**Net Load (MWh)**
$$ \text{Load}_{Net} = \text{Load}_{Actual} - \text{Gen}_{Effective} $$

**REC Cost (USD)**
Cost to buy RECs when in deficit ($\text{Load}_{Net} > 0$).
$$ \text{Cost} = -\text{Load}_{Net} \times \text{Price}_{REC} $$

**REC Revenue (USD)**
Revenue from selling RECs when in surplus ($\text{Load}_{Net} < 0$).
$$ \text{Revenue} = -\text{Load}_{Net} \times \text{Price}_{REC} $$
*(Note: $-\text{Load}_{Net}$ is positive when in surplus)*

**Scarcity Pricing Multipliers**
If enabled, the Base REC Price is multiplied based on month and hour:
*   **Cat 6 (Critical Scarcity):** 2.0x (Dec-Feb, 18:00-20:00)
*   **Cat 5 (Winter Morning):** 1.4x (Dec-Feb, 06:00-09:00)
*   **Cat 4 (Evening Peak):** 1.2x (Other Evenings 17:00-21:00)
*   **Cat 3 (Shoulder):** 1.0x (07:00-10:00 & 15:00-18:00)
*   **Cat 2 (Typical Mid-day):** 0.75x (Nov-Feb, 10:00-15:00)
*   **Cat 1 (Super-abundant):** 0.45x (Mar-Oct, 10:00-15:00)
