import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
st.title('Manzeleya Restocking Decision App')
st.write('This app predicts if a product is worth restocking based on the dataset.')

# Load the dataset
dataset_path = 'Manzeleya Dataset.xlsx'
df = pd.read_excel(dataset_path)

# Display dataset preview
st.write('Dataset preview:')
st.write(df.head())

# Handling missing values
df.fillna(0, inplace=True)

# Feature Engineering: Create a binary target variable (e.g., 'restock' column)
df['restock'] = np.where(df['ordered_item_quantity'] > df['ordered_item_quantity'].mean(), 1, 0)

# Select features and target
features = df[['product_price', 'ordered_item_quantity']]  # You can add more relevant features
target = df['restock']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Model Accuracy: {accuracy:.2f}')
st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))

# Product Selection
st.write('Select a product to determine if it is worth restocking:')
product_name = st.selectbox('Choose a product', df['product_title'].unique())

# Get the selected product's data
product_data = df[df['product_title'] == product_name]
product_features = product_data[['product_price', 'ordered_item_quantity']]

# Prediction
if st.button('Check Restocking Worthiness'):
    restock_prediction = model.predict(product_features)
    restock_probability = model.predict_proba(product_features)[:, 1]

    sell_probability = np.mean(restock_probability)  # Probability of a product to be sold

    if restock_prediction[0] == 1:
        st.write(f'The model suggests restocking {product_name}.')
    else:
        st.write(f'The model suggests not restocking {product_name}.')

    st.write(f'Restocking worthiness: {restock_probability[0]:.2f}')
    st.write(f'Probability of selling the product: {sell_probability:.2f}')

# Detect Most Sold and Least Sold Items
st.write('Most Sold and Least Sold Products')

# Calculate total quantity sold for each product
total_sales = df.groupby('product_title')['ordered_item_quantity'].sum().reset_index()

# Sort products by total quantity sold
most_sold = total_sales.sort_values(by='ordered_item_quantity', ascending=False).head(10)
least_sold = total_sales.sort_values(by='ordered_item_quantity', ascending=True).head(10)

product_sales = df.groupby('product_price')['ordered_item_quantity'].sum().reset_index()

# Scatter plot: Sales by Product Price
st.write('Sales by Product Price')
fig, ax = plt.subplots()

# Scatter plot
ax.scatter(product_sales['product_price'], product_sales['ordered_item_quantity'], label='Data Points')

# Add a trend line
X = product_sales['product_price'].values.reshape(-1, 1)
y = product_sales['ordered_item_quantity'].values
model = LinearRegression()
model.fit(X, y)
trend_line = model.predict(X)
ax.plot(product_sales['product_price'], trend_line, color='red', linewidth=2, label='Trend Line')

# Labels and title
ax.set_xlabel('Product Price')
ax.set_ylabel('Total Quantity Sold')
ax.set_title('Sales by Product Price')
ax.legend()

# Display plot in Streamlit
st.pyplot(fig)

# Convert dates to datetime if not already done
df['random_date'] = pd.to_datetime(df['random_date'])

# Extract month and year for grouping
df['year_month'] = df['random_date'].dt.to_period('M')

# Group by year_month and calculate total quantity sold
monthly_sales = df.groupby('year_month')['ordered_item_quantity'].sum().reset_index()

# Line plot: Sales by Month
st.write('Monthly Sales Analysis')
fig, ax = plt.subplots()
ax.plot(monthly_sales['year_month'].astype(str), monthly_sales['ordered_item_quantity'], marker='o')
ax.set_xlabel('Month')
ax.set_ylabel('Total Quantity Sold')
ax.set_title('Sales by Month')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
st.pyplot(fig)
customer_sales = df.groupby('customer_name')['ordered_item_quantity'].sum().reset_index()

# Sort customers by total quantity purchased and select the top 10
top_customers = customer_sales.sort_values(by='ordered_item_quantity', ascending=False).head(10)

# Bar chart: Top 10 Customers by Purchase Volume
st.write('Top 10 Customers by Purchase Volume')
fig, ax = plt.subplots()
ax.barh(top_customers['customer_name'], top_customers['ordered_item_quantity'], color='skyblue')
ax.set_xlabel('Total Quantity Purchased')
ax.set_ylabel('Customer Name')
ax.set_title('Top 10 Customers by Purchase Volume')
ax.invert_yaxis()  # Invert y-axis to have the top customer at the top
st.pyplot(fig)
# Display the charts
st.write('Top 10 Most Sold Products:')
st.bar_chart(most_sold.set_index('product_title'))

st.write('Top 10 Least Sold Products:')
st.bar_chart(least_sold.set_index('product_title'))

# Save the model (optional)
if st.button('Save Model'):
    joblib.dump(model, 'restocking_model.pkl')
    st.write('Model saved as restocking_model.pkl')
