from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
# 加载模型
model = load_model('my_ai_model')
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    # 返回预测结果
    return jsonify({
        'success': True,
        'prediction': float(prediction[0][0])  # 将预测值转换为浮点数返回
    })
if __name__ == '__main__':
    app.run(debug=True)


