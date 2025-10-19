import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
from sklearn.calibration import CalibratedClassifierCV
st.markdown('<h2 style="font-size:20px;">XGBoost Model for Postoperative Portal Vein Thrombosis​</h2>', unsafe_allow_html=True)

if 'age_valid' not in st.session_state:
    st.session_state.age_valid = True
Age = st.number_input("Age (Years):",
    #min_value=18,      # 最小值
    #max_value=85,      # 最大值
    value=18,          # 默认值（可选，默认为 min_value）
    step=1,            
    help="Must be 18-85 years")
if Age < 18 or Age > 85:
    st.error("Value must be between 18 and 85 yesrs.")
Pre_D_Dimer = st.number_input("Preoperative D-dimer (μg/mL):",
    #min_value=0.00,      # 最小值
    #max_value=10.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,           
    help="Must be 0-10 μg/mL")
if Pre_D_Dimer > 10.00:
    st.error("Value must be between <10μg/mL")
D_Dimer_D3 = st.number_input("D-dimer on Postoperative Day 3 (μg/mL):",
   #min_value=0.00,      # 最小值
    #max_value=10.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,            
    help="Must be 0-10 μg/mL")
if D_Dimer_D3 > 10.00:
    st.error("Value must be between <10μg/mL.")
NET = st.number_input("Postoperative proportion of NETs:",
   #min_value=0.00,      # 最小值
    #max_value=10.00,      # 最大值
    value=0.00,          # 默认值（可选，默认为 min_value）
    step=0.01,           
    help="Must be 0-10")
if NET > 10.00:
    st.error("Value must be between <10")
CA199 = st.number_input("Preoperative CA19-9 (U/mL):",
    #min_value=0.00,      # 最小值
    #max_value=20000.00,      # 最大值
    value=0.0,          # 默认值（可选，默认为 min_value）
    step=0.1,           
    help="Must be 0.0-20000.0 (U/mL)")
if CA199  > 20000.0:
    st.error("Value must be between < 20000.0(10^9/L).")
Operation_duration = st.number_input("Operation Duration (h):",
    #min_value=60,      # 最小值
    #max_value=240,      # 最大值
    value=1.0,          # 默认值（可选，默认为 min_value）
    step=0.1,            
    help="Must be 1.0-18.0 h")
if Operation_duration < 1.0 or Operation_duration > 18.0:
    st.error("Value must be between 1.0 and 18.0 h")
PVR = st.selectbox('Portal Venous Resection', ['non-PVR',"End-to-End Anastomosis",'Replacement'])
PVRmap = {'non-PVR': 0, 'End-to-End Anastomosis': 1, 'Replacement': 2}
PVR = PVRmap[PVR]

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    XGB = joblib.load("XGB.pkl")
    calibrated = joblib.load("calibrated.pkl")
    scaler = joblib.load("scaler.pkl")
    st.write(type(XGB))
    
    # Store inputs into dataframe
    input_numerical = np.array([Age,CA199,D_Dimer_D3,NET,Operation_duration,PVR,Pre_D_Dimer]).reshape(1, -1)
    feature_names  = ["Age","CA199","D_Dimer_D3","NET","Operation_duration","PVR","Pre_D_Dimer"]
    input_numericalyuan = pd.DataFrame(input_numerical, columns=feature_names)
    input_numerical = pd.DataFrame(input_numerical, columns=feature_names)
    input_numerical[["Age","CA199","D_Dimer_D3","NET","Operation_duration","Pre_D_Dimer"]] = scaler.transform(input_numerical[["Age","CA199","D_Dimer_D3","NET","Operation_duration","Pre_D_Dimer"]])


    prediction_proba = calibrated.predict_proba(input_numerical)[:, 1]
    prediction_proba = (prediction_proba * 100).round(2)
    st.markdown("## **Prediction Probabilities (%)**")
    for prob in prediction_proba:
        if prob < 1.00:  
            st.markdown(f"**<1%**")
        else:
            st.markdown(f"**{prob:.2f}%**")

  
    explainer = shap.TreeExplainer(XGB.get_booster())
    shap_values = explainer.shap_values(input_numerical)
    
    st.write("### SHAP Value Force Plot")
    shap.initjs()
    force_plot_visualizer = shap.plots.force(
        explainer.expected_value, shap_values, input_numericalyuan)
    shap.save_html("force_plot.html", force_plot_visualizer)

    with open("force_plot.html", "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

    components.html(html_content, height=400)
