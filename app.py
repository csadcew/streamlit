import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
h=""
# Load the fetal state model
# model = joblib.load(r'./BP-ANN.pkl')
model = joblib.load(h+'BP-ANN.pkl')
# Define feature names
feature_names = [
    "Adhesiveness", "Resilience", "Aw", "L*", "a*", "b*", "C*", "S3", "S9",
    "S10", "S12", "Thr", "Met", "Lys", "His", "Pro"]

data = pd.read_csv(h+"data6.csv").iloc[:, :-1].values
# Streamlit user interface
st.title("Meat adulteration predictor")
# Input features
Adhesiveness = st.number_input("Adhesiveness:", min_value=0, max_value=200, value=120)
Resilience = st.number_input("Resilience:", min_value=0, max_value=10, value=0)
Aw = st.number_input("Aw:", value=0)
L_value = st.number_input("L*:", value=0)
a_value = st.number_input("a*:", value=0)
b_value = st.number_input("b*:", value=0)
C_value = st.number_input("C*:", value=0)
S3 = st.number_input("S3:", value=0)
S9 = st.number_input("S9:", value=0)
S10 = st.number_input("S10:", value=0)
S12 = st.number_input("S12:", value=0)
Thr = st.number_input("Thr:", value=0)
Met = st.number_input("Met:", value=0)
Lys = st.number_input("Lys:", value=0)
His = st.number_input("His:", value=0)
Pro = st.number_input("Pro:", value=0)

# Collect input values into a list
feature_values = [Adhesiveness, Resilience, Aw, L_value, a_value, b_value, C_value, S3, S9, S10, S12, Thr, Met, Lys,
                  His, Pro]

# Convert feature values into a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # Predict class and probabilities using DataFrame
    predicted_class = np.argmax(model.predict(features_df)[0])
    predicted = model.predict(features_df)[0]
    l = ["0%", "10%", "20%", "50%", "90%", "100%"]
    # Display prediction results
    st.write(f"**Predicted Class:** {l[predicted_class]}")
    st.write(f"**Prediction Probabilities:** {predicted}")
    print(predicted_class)
    # Generate advice based on prediction results
    probability = predicted[predicted_class] * 100

    if predicted_class == 5:
        advice = (
            f"According to our model, the Patty is in an adulterated state. "
            f"The model predicts that the Patty has a {probability:.1f}% probability of being adulterated. "
        )
    elif predicted_class == 0:
        advice = (
            f"According to our model, the Patty is in a normal state. "
            f"The model predicts that the Patty has a {probability:.1f}% probability of being normal. "
        )
    else:
        advice = (
            f"According to our model, the Patty is in a suspicious state. "
            f"The model predicts that the Patty has a {probability:.1f}% probability of being suspicious. "
        )
    st.write(advice)
    explainer = shap.KernelExplainer(model.predict, data[:50])  # 使用一部分数据作为背景数据
    # 计算 SHAP 值
    shap_values = explainer.shap_values(features_df.values)
    print(shap_values)
    shap_value = shap_values[0, :, predicted_class]  # 提取该类别的 SHAP 值
    base_value = explainer.expected_value[0]  # 基线值
    # 绘制瀑布图
    shap.waterfall_plot(shap.Explanation(values=shap_value,
                                         base_values=base_value,
                                         data=features_df.values[0],
                                         feature_names=feature_names))
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")
