import shap
import tensorflow as tf
import numpy as np

# 生成一些随机数据并训练模型
data = np.random.rand(100, 16)  # 假设数据有16个特征
labels = np.random.randint(6, size=100)  # 六分类
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(16,)),
    tf.keras.layers.Dense(6, activation="softmax")  # 六个输出类别
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(data, labels, epochs=10, verbose=0)

# 创建一个 SHAP Explainer 对象
explainer = shap.KernelExplainer(model.predict, data[:50])  # 使用一部分数据作为背景数据

# 选择一个要解释的样本
sample_to_explain = data[0:1]  # 选择第一个样本

# 计算 SHAP 值
shap_values = explainer.shap_values(sample_to_explain)

# 选择要绘制的类别，这里以类别 0 为例
class_index = 0  # 修改为你希望查看的类别索引
shap_value = shap_values[0, :, class_index]  # 提取该类别的 SHAP 值
base_value = explainer.expected_value[0]  # 基线值

# 绘制瀑布图
shap.waterfall_plot(shap.Explanation(values=shap_value,
                                     base_values=base_value,
                                     data=sample_to_explain[0],
                                     feature_names=[f'Feature {i}' for i in range(data.shape[1])]))
