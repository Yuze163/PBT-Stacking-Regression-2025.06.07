# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:49:14 2025

@author: 86185
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文和负号，这里以微软雅黑为例
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 加载模型
model_path = "stacking_regressor_model.pkl"
stacking_regressor = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")
st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。""")

# 定义特征范围和类型
feature_ranges = {
    "Resilience": {"type": "numerical", "min": 6, "max": 36, "default": 18},
    "Depression": {"type": "numerical", "min": 0, "max": 3, "default": 3},
    "Anxiety": {"type": "numerical", "min": 0, "max": 3, "default": 3},
    "Family_support": {"type": "numerical", "min": 0, "max": 10, "default": 5},
    "Age": {"type": "numerical", "min": 21, "max": 63, "default": 21},
    "Occupation": {"type": "categorical", "options": ["Full-time job", "Part-time job"]},
    "Method_of_delivery": {"type": "categorical", "options": ["Vaginal delivery", "Cesarean section"]},
    "Marital_status": {"type": "categorical", "options": ["Married", "Unmarried"]},
    "Educational_degree": {"type": "categorical", "options": ["Associate degree or below", "Bachelor's degree or above"]},
    "Average_monthly_household_income": {"type": "categorical", "options": ["Average monthly household income less than or equal to 5000 yuan", "Average monthly household income greater than 5000 yuan"]},
    "Medical_insurance": {"type": "categorical", "options": ["No", "Yes"]},
    "Mode_of_conception": {"type": "categorical", "options": ["Natural conception", "Assisted conception"]},
    "Pregnancy_complications": {"type": "categorical", "options": ["Yes", "No"]},
    "Breastfeeding": {"type": "categorical", "options": ["Yes", "No"]},
    "Rooming_in": {"type": "categorical", "options": ["Yes", "No"]},
    "Planned_pregnancy": {"type": "categorical", "options": ["Yes", "No"]},
    "Intrapartum_pain": {"type": "numerical", "min": 0, "max": 10, "default": 5},
    "Postpartum_pain": {"type": "numerical", "min": 0, "max": 10, "default": 5}
}

# 英文特征名称
feature_names = [
    "Resilience", "Depression", "Anxiety", "Family_support", "Age", "Occupation", "Method_of_delivery",
    "Marital_status","Educational_degree","Average_monthly_household_income","Medical_insurance",
    "Mode_of_conception","Pregnancy_complications","Breastfeeding","Rooming_in","Planned_pregnancy",
    "Intrapartum_pain","Postpartum_pain"
]

# 动态生成输入项
st.sidebar.header("变量输入区域")
st.sidebar.write("请输入变量值：")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("预测"):

    # SHAP 解释器 - 修正为使用 stacking_regressor
    explainer = shap.TreeExplainer(stacking_regressor)
    shap_values = explainer.shap_values(features)

    # 获取基础值和第一个样本的 SHAP 值
    if isinstance(explainer.expected_value, list):
        base_value = explainer.expected_value[0]  # 如果是列表，取第一个元素
    else:
        base_value = explainer.expected_value  # 如果是单个值，直接使用

    shap_values_sample = shap_values[0]

    # 创建SHAP瀑布图，确保中文显示
    plt.figure(figsize=(6, 4))  # 设置图形尺寸为6x4英寸
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_sample,
            base_values=base_value,
            data=features[0],
            feature_names=feature_names
        ),
        max_display=10  # 限制显示的特征数量
    )

    # 保存SHAP瀑布图并展示
    plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=600)
    st.image("shap_waterfall_plot.png")
        
        
        
