import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AI Crop Recommendation India",
    page_icon="🌾",
    layout="wide"
)

# ---------------- BACKGROUND IMAGE ----------------

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.block-container {
    background-color: rgba(255, 255, 255, 0.90);
    padding: 2rem;
    border-radius: 15px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- LOAD DATA & TRAIN MODEL (Cached) ----------------

@st.cache_resource
def load_model():
    data = pd.read_csv("Crop_recommendation.csv")
    X = data.drop("label", axis=1)
    y = data["label"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_columns = load_model()

# ---------------- STATE + DISTRICT DATA ----------------

india_districts = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubli"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi"],
    "West Bengal": ["Kolkata", "Siliguri", "Durgapur"]
}

state_data = {
    state: {"temp": 28, "rain": 110, "humidity": 65}
    for state in india_districts.keys()
}

# ---------------- SOIL TYPE ----------------

soil_map = {
    "Black Soil": {"N": 80, "P": 60, "K": 70},
    "Red Soil": {"N": 50, "P": 40, "K": 50},
    "Sandy Soil": {"N": 30, "P": 20, "K": 30},
    "Clay Soil": {"N": 70, "P": 50, "K": 60},
    "Alluvial Soil": {"N": 65, "P": 55, "K": 65}
}

# ---------------- FERTILIZER SUGGESTION ----------------

fertilizer_map = {
    "rice": "Use Urea and DAP for better nitrogen support.",
    "wheat": "Apply NPK fertilizer (20-20-0) during early growth stage.",
    "maize": "Use Urea and Potash fertilizers.",
    "cotton": "Apply NPK (30-15-15) and organic compost.",
    "sugarcane": "Use high nitrogen fertilizers and farmyard manure.",
    "banana": "Apply Potassium-rich fertilizer.",
    "mango": "Use organic compost and balanced NPK.",
    "grapes": "Apply Super Phosphate and Potash."
}

# ---------------- UI HEADER ----------------

st.markdown("<h1 style='text-align:center;color:green;'>🌾 AI Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Smart Agriculture Support for Indian Farmers 🇮🇳</h4>", unsafe_allow_html=True)

st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Choose User Type",
    ["🏠 Home", "👨‍🌾 General Farmer", "🧪 Soil-Test Farmer"]
)

# ---------------- HOME ----------------

if menu == "🏠 Home":
    st.info("""
    ✅ Select user type  
    ✅ Enter required details  
    ✅ Click Predict  
    ✅ Get Crop + Fertilizer + Yield  

    Supporting Indian Farmers 🇮🇳
    """)

# ---------------- GENERAL FARMER ----------------

elif menu == "👨‍🌾 General Farmer":

    st.subheader("Farmer Details")

    name = st.text_input("Farmer Name")
    farm_size = st.number_input("Farm Size (Acres)", min_value=0.0)

    state = st.selectbox("Select State", list(india_districts.keys()))
    district = st.selectbox("Select District", india_districts[state])

    soil_type = st.selectbox("Select Soil Type", list(soil_map.keys()))

    season = st.selectbox(
        "Select Current Farming Season",
        [
            "Kharif (June - October | Rainy Season)",
            "Rabi (November - March | Winter Season)",
            "Summer (April - May | Hot Season)"
        ]
    )

    if st.button("🌾 Predict Best Crop"):

        if not name or farm_size <= 0:
            st.warning("Please enter farmer name and farm size.")
        else:
            climate = state_data[state]
            soil = soil_map[soil_type]

            input_data = pd.DataFrame(
                [[
                    soil["N"],
                    soil["P"],
                    soil["K"],
                    climate["temp"],
                    climate["humidity"],
                    6.5,
                    climate["rain"]
                ]],
                columns=feature_columns
            )

            prediction = model.predict(input_data)
            crop = prediction[0]

            fertilizer = fertilizer_map.get(
                crop.lower(),
                "Use balanced NPK fertilizer and organic compost."
            )

            estimated_yield = round(farm_size * 2.5, 2)

            st.success(f"""
👨‍🌾 Farmer: {name}  
📍 Location: {district}, {state}  

🌾 Recommended Crop: **{crop.upper()}**

🧪 Fertilizer Advice:
{fertilizer}

📦 Estimated Yield:
Approximately **{estimated_yield} tons**
""")

            st.balloons()

# ---------------- SOIL TEST FARMER ----------------

elif menu == "🧪 Soil-Test Farmer":

    st.subheader("Enter Soil & Environmental Values")

    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH Value", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

    if st.button("🌾 Predict Crop"):

        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=feature_columns
        )

        prediction = model.predict(input_data)
        crop = prediction[0]

        fertilizer = fertilizer_map.get(
            crop.lower(),
            "Use balanced NPK fertilizer and organic compost."
        )

        st.success(f"""
🌾 Recommended Crop: **{crop.upper()}**

🧪 Fertilizer Advice:
{fertilizer}
""")

        st.balloons()
