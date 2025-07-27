import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Display scikit-learn version
print(sklearn.__version__)

# Load only Decision Tree model
def load_model():
    model = pickle.load(open('dt_dep.pkl', 'rb'))
    return model

# Main Streamlit App
def main():
    st.set_page_config(page_title="Flight Satisfaction Predictor", layout="wide")

    st.markdown("<h1 style='text-align: center; color: #3366cc;'>✈️ Flight Passenger Satisfaction Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.header("🔧 Model Info")
    model = load_model()

    st.sidebar.markdown("✅ Model Loaded: **`DecisionTree`**")

    # Model info
    with st.expander("📘 Trained Model Configuration", expanded=False):
        st.write(model)

    if model is None:
        st.error("⚠️ Model not available.")
        return

    # Passenger Input Section
    st.markdown("## 🧑‍✈️ Passenger Information")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age', min_value=18, max_value=70, value=30)
            flight_distance = st.number_input('Flight Distance', min_value=100, max_value=5000, value=1000)
            inflight_wifi = st.slider('Inflight Wifi Service', 1, 5, 3)
            dep_arr_time = st.slider('Departure/Arrival Time Convenient', 1, 5, 3)
            online_booking = st.slider('Ease of Online Booking', 1, 5, 3)
            gate_location = st.slider('Gate Location', 1, 5, 3)
            food_drink = st.slider('Food and Drink', 1, 5, 3)
            online_boarding = st.slider('Online Boarding', 1, 5, 3)
            seat_comfort = st.slider('Seat Comfort', 1, 5, 3)
            inflight_entertainment = st.slider('Inflight Entertainment', 1, 5, 3)
            onboard_service = st.slider('On-board Service', 1, 5, 3)
            leg_room_service = st.slider('Leg Room Service', 1, 5, 3)
            baggage_handling = st.slider('Baggage Handling', 1, 5, 3)
            checkin_service = st.slider('Check-in Service', 1, 5, 3)
            inflight_service = st.slider('Inflight Service', 1, 5, 3)
            cleanliness = st.slider('Cleanliness', 1, 5, 3)
            dep_delay = st.number_input('Departure Delay (min)', 0, 1000, 0)
            arr_delay = st.number_input('Arrival Delay (min)', 0, 1000, 0)

        with col2:
            st.markdown("### Demographics & Flight Class")
            gender = st.selectbox('Gender', ['Male', 'Female'])
            customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'Disloyal Customer'])
            travel_type = st.selectbox('Type of Travel', ['Business travel', 'Personal Travel'])
            travel_class = st.selectbox('Class', ['Eco', 'Eco Plus', 'Business'])

    # Manual encoding
    gender_male = 1 if gender == 'Male' else 0
    customer_disloyal = 1 if customer_type == 'Disloyal Customer' else 0
    travel_personal = 1 if travel_type == 'Personal Travel' else 0
    class_eco = 1 if travel_class == 'Eco' else 0
    class_eco_plus = 1 if travel_class == 'Eco Plus' else 0

    input_data = {
        'Age': age,
        'Flight Distance': flight_distance,
        'Inflight wifi service': inflight_wifi,
        'Departure/Arrival time convenient': dep_arr_time,
        'Ease of Online booking': online_booking,
        'Gate location': gate_location,
        'Food and drink': food_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': inflight_entertainment,
        'On-board service': onboard_service,
        'Leg room service': leg_room_service,
        'Baggage handling': baggage_handling,
        'Checkin service': checkin_service,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Departure Delay in Minutes': dep_delay,
        'Arrival Delay in Minutes': arr_delay,
        'Gender_Male': gender_male,
        'Customer Type_disloyal Customer': customer_disloyal,
        'Type of Travel_Personal Travel': travel_personal,
        'Class_Eco': class_eco,
        'Class_Eco Plus': class_eco_plus
    }

    input_df = pd.DataFrame([input_data])
    
    st.markdown("## 📊 Model Input Preview")
    st.dataframe(input_df, use_container_width=True)

    # Prediction
    st.markdown("## 🎯 Prediction")
    if st.button("🚀 Predict Satisfaction"):
        prediction = model.predict(input_df)

        try:
            prob = model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame({
                'Satisfaction Level': model.classes_,
                'Probability': prob
            }).sort_values('Probability', ascending=False)

            st.success(f"✅ Predicted Satisfaction: **{prediction[0]}**")
            st.dataframe(prob_df)
            st.bar_chart(prob_df.set_index('Satisfaction Level'))

        except Exception as e:
            st.success(f"✅ Predicted Satisfaction: **{prediction[0]}**")
            st.info("ℹ️ Model does not support probability prediction.")
            st.exception(e)

# Run the app
if __name__ == '__main__':
    main()
