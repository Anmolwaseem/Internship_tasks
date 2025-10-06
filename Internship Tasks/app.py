import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("superstore.csv", encoding="latin1")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    return df

df = load_data()

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")
region = st.sidebar.multiselect("Select Region", df["Region"].unique())
category = st.sidebar.multiselect("Select Category", df["Category"].unique())
sub_category = st.sidebar.multiselect("Select Sub-Category", df["Sub-Category"].unique())

# Apply filters
filtered_df = df.copy()
if region:
    filtered_df = filtered_df[filtered_df["Region"].isin(region)]
if category:
    filtered_df = filtered_df[filtered_df["Category"].isin(category)]
if sub_category:
    filtered_df = filtered_df[filtered_df["Sub-Category"].isin(sub_category)]

# ---------------- KPIs ----------------
st.title(" Business Dashboard - Global Superstore")
st.subheader("Key Performance Indicators (KPIs)")

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
top_customers = filtered_df.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(5)

col1, col2 = st.columns(2)
col1.metric("Total Sales", f"${total_sales:,.2f}")
col2.metric("Total Profit", f"${total_profit:,.2f}")

# ---------------- Charts ----------------
st.subheader(" Sales vs Profit by Category")
fig, ax = plt.subplots()
sns.barplot(x="Category", y="Sales", data=filtered_df, estimator=sum, ci=None, ax=ax)
st.pyplot(fig)

st.subheader("💡 Top 5 Customers by Sales")
st.bar_chart(top_customers)
