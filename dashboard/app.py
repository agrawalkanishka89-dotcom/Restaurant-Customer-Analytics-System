import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from datetime import date, datetime
from streamlit_autorefresh import st_autorefresh


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Restaurant Customer Analytics",
    page_icon="🍽️",
    layout="wide"
)

st_autorefresh(
    interval=5000,
    key="restaurant_dashboard_refresh"
)

st.title("🍽️ Restaurant Customer Analytics")
st.caption(
    "Customer Entry/Exit Monitoring | "
    f"Last refreshed: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)
st.divider()


# =========================================================
# 2. SETTINGS AND CSV LOCATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "data" / "customer_records.csv"
RESTAURANT_CAPACITY = 50


# =========================================================
# 3. LOAD CSV SAFELY
# =========================================================

if not CSV_FILE.exists():
    st.error("customer_records.csv file not found.")
    st.code(str(CSV_FILE))
    st.stop()

try:
    if CSV_FILE.stat().st_size == 0:
        st.warning("CSV file is empty. Waiting for customer records.")
        st.stop()

    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8-sig",
        dtype=str
    )

except pd.errors.EmptyDataError:
    st.warning("CSV file is empty.")
    st.stop()

except PermissionError:
    st.error("CSV file is currently being used by another application.")
    st.stop()

except Exception as error:
    st.error(f"CSV reading error: {error}")
    st.stop()


# =========================================================
# 4. NORMALIZE AND VALIDATE COLUMNS
# =========================================================

df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

column_mapping = {
    "customer_id": "Customer_ID",
    "customerid": "Customer_ID",
    "tracking_id": "Tracking_ID",
    "trackingid": "Tracking_ID",
    "id": "Tracking_ID",
    "event": "Event",
    "date": "Date",
    "time": "Time",
    "gender": "Gender"
}

df = df.rename(columns=column_mapping)

required_columns = ["Tracking_ID", "Event", "Date", "Time"]
missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    st.error("CSV file columns are incorrect.")
    st.write("Available columns:", df.columns.tolist())
    st.write("Missing columns:", missing_columns)
    st.info("Required columns: Tracking_ID, Event, Date, Time")
    st.stop()

if "Customer_ID" not in df.columns:
    df["Customer_ID"] = ""

if "Gender" not in df.columns:
    df["Gender"] = "Unknown"

df = df[
    ["Customer_ID", "Tracking_ID", "Event", "Date", "Time", "Gender"]
].copy()


# =========================================================
# 5. CLEAN DATA AND CREATE DATETIME
# =========================================================

for column in ["Customer_ID", "Tracking_ID", "Event", "Date", "Time", "Gender"]:
    df[column] = df[column].fillna("").astype(str).str.strip()

df["Event"] = df["Event"].str.upper()

df["Gender"] = df["Gender"].str.capitalize()
df.loc[~df["Gender"].isin(["Male", "Female"]), "Gender"] = "Unknown"

# Support common date formats used by the project CSV.
date_text = df["Date"]

parsed_dates = pd.to_datetime(
    date_text,
    format="%Y-%m-%d",
    errors="coerce"
)

missing_dates = parsed_dates.isna()
parsed_dates.loc[missing_dates] = pd.to_datetime(
    date_text.loc[missing_dates],
    format="%d-%m-%Y",
    errors="coerce"
)

missing_dates = parsed_dates.isna()
parsed_dates.loc[missing_dates] = pd.to_datetime(
    date_text.loc[missing_dates],
    format="%d/%m/%Y",
    errors="coerce"
)

parsed_times = pd.to_timedelta(df["Time"], errors="coerce")
df["DateTime"] = parsed_dates + parsed_times

invalid_records = int(df["DateTime"].isna().sum())

if invalid_records:
    st.warning(f"{invalid_records} rows with invalid Date/Time were ignored.")

df = df.dropna(subset=["DateTime"]).copy()
df = df[df["Event"].isin(["ENTRY", "EXIT"])].copy()

if df.empty:
    st.error("No valid ENTRY/EXIT records are available in the CSV.")
    st.stop()


# =========================================================
# 6. DATE FILTER
# =========================================================

st.subheader("📅 Filter Customer Data")

available_min_date = df["DateTime"].dt.date.min()
available_max_date = df["DateTime"].dt.date.max()

# Do not extend the selectable range beyond dates present in the CSV.
selected_dates = st.date_input(
    "Select date range",
    value=(available_min_date, available_max_date),
    min_value=available_min_date,
    max_value=available_max_date
)

if isinstance(selected_dates, (tuple, list)):
    if len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date = end_date = available_min_date
else:
    start_date = end_date = selected_dates

if start_date > end_date:
    start_date, end_date = end_date, start_date

filtered_df = df[
    df["DateTime"].dt.date.between(start_date, end_date)
].copy()

st.caption(f"Showing records from {start_date} to {end_date}")
st.divider()


# =========================================================
# 7. ENTRY / EXIT DATA AND RESTAURANT OVERVIEW
# =========================================================

entries = filtered_df[filtered_df["Event"] == "ENTRY"].copy()
exits = filtered_df[filtered_df["Event"] == "EXIT"].copy()

total_entries = len(entries)
total_exits = len(exits)

# Occupancy is calculated from the complete CSV, not just selected dates.
all_entries_count = int((df["Event"] == "ENTRY").sum())
all_exits_count = int((df["Event"] == "EXIT").sum())

current_occupancy = max(0, all_entries_count - all_exits_count)
available_capacity = max(0, RESTAURANT_CAPACITY - current_occupancy)

st.subheader("📌 Restaurant Overview")

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

overview_col1.metric("Total Entries", total_entries)
overview_col2.metric("Total Exits", total_exits)
overview_col3.metric("Estimated Current Occupancy", current_occupancy)
overview_col4.metric("Estimated Available Capacity", available_capacity)

st.caption(
    f"Configured restaurant capacity: {RESTAURANT_CAPACITY}. "
    "Occupancy is estimated as all recorded ENTRY events minus all EXIT events."
)
st.divider()


# =========================================================
# 8. GENDER ANALYTICS
# =========================================================

st.subheader("👥 Gender Analytics")

gender_entries = entries[entries["Gender"].isin(["Male", "Female", "Unknown"])]

male_entries = int((gender_entries["Gender"] == "Male").sum())
female_entries = int((gender_entries["Gender"] == "Female").sum())
unknown_entries = int((gender_entries["Gender"] == "Unknown").sum())
known_gender_entries = male_entries + female_entries

gender_col1, gender_col2, gender_col3, gender_col4 = st.columns(4)

gender_col1.metric("Male Entries", male_entries)
gender_col2.metric("Female Entries", female_entries)
gender_col3.metric("Unknown Gender Entries", unknown_entries)
gender_col4.metric("Known Gender Entries", known_gender_entries)

if known_gender_entries > 0:
    gender_chart_data = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Entries": [male_entries, female_entries]
    })
    gender_chart_data = gender_chart_data[gender_chart_data["Entries"] > 0]

    gender_left, gender_right = st.columns(2)

    with gender_left:
        gender_pie = px.pie(
            gender_chart_data,
            names="Gender",
            values="Entries",
            hole=0.45,
            title="Male vs Female Entry Distribution"
        )
        st.plotly_chart(gender_pie, use_container_width=True)

    with gender_right:
        gender_bar = px.bar(
            gender_chart_data,
            x="Gender",
            y="Entries",
            text="Entries",
            title="Gender-wise Customer Entries"
        )
        gender_bar.update_layout(
            xaxis_title="Gender",
            yaxis_title="Number of ENTRY records"
        )
        st.plotly_chart(gender_bar, use_container_width=True)
else:
    st.info("No Male/Female gender predictions are available for this date range.")

st.caption(
    "Gender figures count ENTRY records, not unique people. "
    "Unknown means gender prediction was unavailable or uncertain."
)
st.divider()


# =========================================================
# 9. HOURLY CUSTOMER ANALYSIS + PEAK HOUR INSIGHT
# =========================================================

st.subheader("📊 Hourly Customer Analysis")

if not entries.empty:
    hourly_data = (
        entries.groupby(entries["DateTime"].dt.hour)
        .size()
        .reindex(range(24), fill_value=0)
        .rename("Customers")
        .reset_index()
    )
    hourly_data.columns = ["Hour", "Customers"]
    hourly_data["Hour_Label"] = hourly_data["Hour"].map(lambda hour: f"{hour:02d}:00")

    busiest_row = hourly_data.loc[hourly_data["Customers"].idxmax()]
    active_hours = hourly_data[hourly_data["Customers"] > 0]

    peak_col1, peak_col2 = st.columns(2)
    peak_col1.metric(
        "🔥 Busiest Recorded Hour",
        busiest_row["Hour_Label"],
        f"{int(busiest_row['Customers'])} entries"
    )

    if not active_hours.empty:
        quietest_row = active_hours.loc[active_hours["Customers"].idxmin()]
        peak_col2.metric(
            "📉 Least Busy Active Hour",
            quietest_row["Hour_Label"],
            f"{int(quietest_row['Customers'])} entries"
        )

    hourly_chart = px.bar(
        hourly_data,
        x="Hour_Label",
        y="Customers",
        text="Customers",
        title="Customer Entry Traffic by Hour"
    )
    hourly_chart.update_layout(
        xaxis_title="Hour",
        yaxis_title="Number of ENTRY records",
        xaxis={"categoryorder": "array", "categoryarray": hourly_data["Hour_Label"].tolist()}
    )
    st.plotly_chart(hourly_chart, use_container_width=True)
else:
    st.info("No customer entries are available for hourly analysis.")

st.divider()


# =========================================================
# 10. DAILY CUSTOMER ANALYSIS
# =========================================================

st.subheader("📅 Daily Customer Analysis")

if not entries.empty:
    daily_data = (
        entries.groupby(entries["DateTime"].dt.date)
        .size()
        .reset_index(name="Customers")
    )
    daily_chart = px.line(
        daily_data,
        x="DateTime",
        y="Customers",
        markers=True,
        text="Customers",
        title="Daily Customer Entry Trend"
    )
    daily_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of ENTRY records"
    )
    st.plotly_chart(daily_chart, use_container_width=True)
else:
    st.info("No daily customer data is available for this date range.")

st.divider()


# =========================================================
# 11. CUSTOMER VISIT FREQUENCY + CUSTOMER VISIT DETAILS
# =========================================================

st.subheader("🔁 Customer Visit Frequency Analysis")

# One ENTRY row is treated as one recorded visit/arrival.
# EXIT rows are deliberately excluded so an entry and its exit
# are not counted as two visits.
identified_entries = entries[
    entries["Customer_ID"].notna()
    & (entries["Customer_ID"].astype(str).str.strip() != "")
].copy()

if not identified_entries.empty:
    visit_frequency = (
        identified_entries.groupby("Customer_ID")
        .agg(
            Total_Visits=("Event", "size"),
            First_Visit=("DateTime", "min"),
            Last_Visit=("DateTime", "max")
        )
        .reset_index()
        .sort_values(
            by=["Total_Visits", "Last_Visit"],
            ascending=[False, False]
        )
    )

    unique_customers = len(visit_frequency)
    repeat_customers = int((visit_frequency["Total_Visits"] > 1).sum())
    one_time_customers = int((visit_frequency["Total_Visits"] == 1).sum())

    visit_col1, visit_col2, visit_col3 = st.columns(3)
    visit_col1.metric("Identified Unique Customers", unique_customers)
    visit_col2.metric("Repeat Customers", repeat_customers)
    visit_col3.metric("One-time Customers", one_time_customers)

    top_customers = visit_frequency.head(15).copy()

    frequency_chart = px.bar(
        top_customers.sort_values("Total_Visits", ascending=True),
        x="Total_Visits",
        y="Customer_ID",
        orientation="h",
        text="Total_Visits",
        title="Top 15 Customers by Recorded Visits"
    )
    frequency_chart.update_layout(
        xaxis_title="Recorded ENTRY visits",
        yaxis_title="Customer ID"
    )
    st.plotly_chart(frequency_chart, use_container_width=True)

    st.subheader("📋 Customer Visit Details")

    visit_details = visit_frequency.copy()
    visit_details["First Visit"] = visit_details["First_Visit"].dt.strftime("%d-%m-%Y %H:%M:%S")
    visit_details["Last Visit"] = visit_details["Last_Visit"].dt.strftime("%d-%m-%Y %H:%M:%S")

    # Display the most frequently recorded gender for each Customer_ID.
    gender_by_customer = (
        identified_entries[identified_entries["Gender"].isin(["Male", "Female"])]
        .groupby("Customer_ID")["Gender"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown")
    )

    visit_details["Gender"] = visit_details["Customer_ID"].map(gender_by_customer).fillna("Unknown")

    visit_details = visit_details.rename(columns={
        "Customer_ID": "Customer ID",
        "Total_Visits": "Total Visits"
    })

    visit_details = visit_details[
        ["Customer ID", "Gender", "First Visit", "Last Visit", "Total Visits"]
    ]

    st.dataframe(
        visit_details,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Visit frequency is calculated by counting ENTRY rows per Customer_ID "
        "within the selected date range. EXIT rows are not counted as visits."
    )
else:
    st.info(
        "No non-blank Customer_ID values are available in ENTRY records "
        "for this date range. Customer-level visit analysis will appear "
        "when the camera system writes Customer_ID values."
    )

blank_id_entries = int(
    (entries["Customer_ID"].astype(str).str.strip() == "").sum()
)

if blank_id_entries:
    st.warning(
        f"{blank_id_entries} ENTRY records have a blank Customer_ID. "
        "These records are included in overall entry analytics but excluded "
        "from unique/repeat customer calculations because their identity is unknown."
    )

st.divider()


# =========================================================
# 12. CUSTOMER ENTRY AND EXIT HISTORY
# =========================================================

st.subheader("📋 Customer Entry and Exit History")

history_df = filtered_df.sort_values(
    by="DateTime",
    ascending=False
).copy()

history_columns = [
    "Customer_ID",
    "Tracking_ID",
    "Event",
    "Date",
    "Time",
    "Gender"
]

history_columns = [column for column in history_columns if column in history_df.columns]

history_display = history_df[history_columns].copy()

st.dataframe(
    history_display,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 13. DOWNLOAD REPORT
# =========================================================

st.divider()
st.subheader("📥 Download Customer Report")

download_data = history_display.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Download Filtered Customer Report",
    data=download_data,
    file_name="restaurant_customer_report.csv",
    mime="text/csv"
)


# =========================================================
# 14. DATA QUALITY / DEBUG INFORMATION
# =========================================================

with st.expander("🔍 CSV Data Information"):
    st.write("CSV file location:")
    st.code(str(CSV_FILE))

    st.write("CSV columns:")
    st.write(df.columns.tolist())

    st.write("Total valid ENTRY/EXIT records:", len(df))
    st.write("Records in selected date range:", len(filtered_df))
    st.write("ENTRY records in selected date range:", total_entries)
    st.write("EXIT records in selected date range:", total_exits)
    st.write("ENTRY records with blank Customer_ID:", blank_id_entries)
    st.write("Available dates:")
    st.write(sorted(df["DateTime"].dt.date.unique().tolist()))
