import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load dataset
df = pd.read_csv("mobile_price_dataset_realistic_cleaned.csv")

# Prepare features and targets
X = df.drop(columns=["price", "price_range"])
y_price = df["price"]
y_range = df["price_range"]

# Train models
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
regressor.fit(X, y_price)
classifier.fit(X, y_range)

# Save models (optional)
joblib.dump(regressor, "regressor_model.pkl")
joblib.dump(classifier, "classifier_model.pkl")

# Streamlit UI
st.set_page_config(page_title="📱 Mobile Price & Category Predictor", layout="centered")
st.title("📱 Mobile Phone Price & Range Predictor")
st.markdown("🔍 Enter the specifications to predict **market price** and **price category**.")

# --- Inputs ---
battery_power = st.slider("Battery Power (mAh)", 500, 6000, 2000)
blue = st.radio("Bluetooth", [0, 1], format_func=lambda x: "Yes" if x else "No")
clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
dual_sim = st.radio("Dual SIM", [0, 1], format_func=lambda x: "Yes" if x else "No")
fc = st.slider("Front Camera (MP)", 0, 64, 8)
four_g = st.radio("4G Support", [0, 1], format_func=lambda x: "Yes" if x else "No")
int_memory = st.slider("Internal Memory (GB)", 2, 512, 64)
m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.5, 0.5)
mobile_wt = st.slider("Weight (g)", 80, 300, 150)
n_cores = st.slider("No. of Cores", 1, 12, 4)
pc = st.slider("Primary Camera (MP)", 0, 108, 12)
px_height = st.slider("Pixel Height", 500, 4000, 1280)
px_width = st.slider("Pixel Width", 500, 4000, 720)
ram = st.slider("RAM (MB)", 512, 12000, 4000)
sc_h = st.slider("Screen Height (cm)", 5, 20, 10)
sc_w = st.slider("Screen Width (cm)", 5, 15, 7)
talk_time = st.slider("Talk Time (hrs)", 2, 30, 10)
touch_screen = st.radio("Touch Screen", [0, 1], format_func=lambda x: "Yes" if x else "No")
wifi = st.radio("Wi-Fi", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Network Type
network = st.radio("Network Type", ["2G", "3G", "4G", "5G"])
network_2G = int(network == "2G")
network_3G = int(network == "3G")
network_4G = int(network == "4G")
network_5G = int(network == "5G")

# Input data
input_data = {
    "battery_power": battery_power,
    "blue": blue,
    "clock_speed": clock_speed,
    "dual_sim": dual_sim,
    "fc": fc,
    "four_g": four_g,
    "int_memory": int_memory,
    "m_dep": m_dep,
    "mobile_wt": mobile_wt,
    "n_cores": n_cores,
    "pc": pc,
    "px_height": px_height,
    "px_width": px_width,
    "ram": ram,
    "sc_h": sc_h,
    "sc_w": sc_w,
    "talk_time": talk_time,
    "touch_screen": touch_screen,
    "wifi": wifi,
    "network_2G": network_2G,
    "network_3G": network_3G,
    "network_4G": network_4G,
    "network_5G": network_5G
}

input_df = pd.DataFrame([input_data])

# Map for price range labels
price_range_labels = {
    0: "0 = Low Cost (₹2,000 – ₹9,000)",
    1: "1 = Medium Cost (₹9,100 – ₹20,000)",
    2: "2 = High Cost (₹21,000 – ₹60,000)",
    3: "3 = Very High Cost (₹61,000 – ₹1,50,000)"
}

# Predict
if st.button("Predict 📊"):
    predicted_price = regressor.predict(input_df)[0]
    predicted_range = classifier.predict(input_df)[0]

    st.success(f"💰 **Estimated Price:** ₹{int(predicted_price):,}")
    st.info(f"🏷️ **Price Category:** {price_range_labels[predicted_range]}")