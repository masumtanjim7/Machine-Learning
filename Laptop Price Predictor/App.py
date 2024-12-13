import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')

# Function to preprocess input features
def preprocess_input(X):
    return X

def predict_price(features):
    processed_features = preprocess_input(features)
    predicted_price = np.exp(model.predict(processed_features))
    return round(predicted_price[0])

# Streamlit app
def main():
    st.title("🔮Laptop Price Predictor🔮")
    st.markdown("""
    Welcome to the **Laptop Price Predictor** app! 🎉👨‍💻🔥
    
    Enter the details of the laptop you're interested in, and let the magic happen! ✨
   
   
    Created By: Md. Maasum Billah (21225103383)
    
    
    """)

    # Sidebar - Input features
    st.sidebar.header('🛠️ Input Features')

    example_input = {
        'Company': 'HP',
        'TypeName': 'Notebook',
        'Ram': 8,
        'Weight': 1.5,
        'HDD': 256,
        'SSD': 512,
        'Gpu brand': 'Nvidia',
        'Touchscreen': 'No',
        'Ips': 'Yes',
        'ppi': 200,
        'Cpu brand': 'Intel Core i7',
        'os': 'Windows'
    }


    company = st.sidebar.selectbox('🏷️ Company', ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI', 'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer', 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'], help='Select the company of the laptop.')
    type_name = st.sidebar.selectbox('📋 Type Name', ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation'], help='Select the type of the laptop.')
    ram = st.sidebar.slider('💾 RAM (GB)', 2, 64, 8, help='Choose the RAM capacity of the laptop.')
    weight = st.sidebar.slider('⚖️ Weight (kg)', 0.5, 5.0, 1.5, step=0.01, help='Choose the weight of the laptop.')
    hdd = st.sidebar.slider('📀 HDD (GB)', 0, 1024, 256, help='Choose the HDD storage capacity of the laptop.')
    ssd = st.sidebar.slider('💽 SSD (GB)', 0, 1024, 512, help='Choose the SSD storage capacity of the laptop.')
    gpu_brand = st.sidebar.selectbox('🎮 GPU Brand', ['Intel', 'AMD', 'Nvidia'], help='Select the GPU brand of the laptop.')
    touchscreen = st.sidebar.radio('🖥️ Touchscreen', ['No', 'Yes'], help='Select whether the laptop has a touchscreen.')
    ips = st.sidebar.radio('👁️ IPS', ['No', 'Yes'], help='Select whether the laptop has IPS display.')
    ppi = st.sidebar.slider('📏 PPI', 0, 400, 200, help='Choose the PPI (Pixels Per Inch) of the laptop display.')
    cpu_brand = st.sidebar.selectbox('🧠 CPU Brand', ['Intel Core i5', 'Intel Core i7', 'AMD Processor', 'Intel Core i3', 'Other Intel Processor'], help='Select the CPU brand of the laptop.')
    os = st.sidebar.selectbox('💻 Operating System', ['Mac', 'Others/No OS/Linux', 'Windows'], help='Select the operating system of the laptop.')

    # Predict button
    if st.sidebar.button('🔮 Predict Price'):
        input_df = pd.DataFrame({'Company': [company],
                                 'TypeName': [type_name],
                                 'Ram': [ram],
                                 'Weight': [weight],
                                 'HDD': [hdd],
                                 'SSD': [ssd],
                                 'Gpu brand': [gpu_brand],
                                 'Touchscreen': [1 if touchscreen == 'Yes' else 0],
                                 'Ips': [1 if ips == 'Yes' else 0],
                                 'ppi': [ppi],
                                 'Cpu brand': [cpu_brand],
                                 'os': [os]})

        predicted_price = predict_price(input_df)

        st.success('💰 Predicted Price: TK-' + str(predicted_price))
        st.subheader("📝 Laptop Details")
        st.write(input_df)

    st.sidebar.subheader("📊 Model Evaluation Metrics")
    st.sidebar.write("**Random Forest Regression Model:**")
    st.sidebar.write("R-squared Score: 0.8835")
    st.sidebar.write("Mean Absolute Error: 0.1592")
    st.sidebar.write("**Gradient Boosting Regression Model:**")
    st.sidebar.write("R-squared Score: 0.8771")
    st.sidebar.write("Mean Absolute Error: 0.1626")

if __name__ == '__main__':
    main()