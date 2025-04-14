###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('Exceed_model_xgb.pkl')
scaler = joblib.load('Exceed_scaler.pkl') 

# Streamlit user interface
st.title("Sirolimus High Exposure (>15 ng/mL) Predictor")

# Define feature names
feature_names = ['Height', 'HDL', 'TC', 'PLT', 'IL10_rs1800896_CT']

Height = st.number_input("Height (cm):", min_value=0, max_value=180, value=120)
HDL = st.number_input("HDL (mmol/L):", min_value=0.00, max_value=3.00, value=1.30)
TC = st.number_input("TC (mmol/L):", min_value=0.00, max_value=10.00, value=4.00)
PLT = st.number_input("PLT (109/L):", min_value=0, max_value=400, value=250)
IL10_rs1800896_CT = st.selectbox("IL-10.rs1800896_CT:", options=[1, 2], format_func=lambda x: 'No' if x == 1 else 'Yes')


# 准备输入特征
feature_values = [Height, HDL, TC, PLT, IL10_rs1800896_CT]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [Height, HDL, TC, PLT]
categorical_features=[IL10_rs1800896_CT]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)

# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=['Height', 'HDL', 'TC', 'PLT'])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)

# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)


if st.button("Predict"):    
    OPTIMAL_THRESHOLD = 0.261
    
    # Predict class and probabilities    
    #predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]
    prob_class1 = predicted_proba[1]  # 类别1的概率

    # 根据最优阈值判断类别
    predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

    # 显示结果（概率形式更直观）
    st.write(f"**High Exposure Probability:** {prob_class1:.1%}")
    st.write(f"**Decision Threshold:** {OPTIMAL_THRESHOLD:.0%} (optimized for clinical utility)")
    st.write(f"**Predicted Class:** {predicted_class} (1: High risk, 0: Low risk)")

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")

    # 创建SHAP解释器
    # 假设 X_train 是用于训练模型的特征数据
    df=pd.read_csv('Exceed_train_lasso.csv',encoding='utf8')
    ytrain=df.Exceed
    x_train=df.drop('Exceed',axis=1)
    from sklearn.preprocessing import StandardScaler
    continuous_cols = [1,4]
    xtrain = x_train.copy()
    scaler = StandardScaler()
    xtrain.iloc[:, continuous_cols] = scaler.fit_transform(x_train.iloc[:, continuous_cols])

    explainer_shap = shap.TreeExplainer(model)
    
    # 获取SHAP值
    shap_values = explainer_shap.shap_values(pd.DataFrame(final_features_df,columns=feature_names))
    
  # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

# Display the SHAP force plot for the predicted class    
    if predicted_class == 1:        
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], original_feature_values, matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], original_feature_values, matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
