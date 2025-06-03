import sys
import os
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化应用
app = Flask(__name__)

def load_trained_model():
    """加载训练好的模型"""
    model_path = os.path.join('models', 'saved_models', 'best_model.h5')
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            logger.info("模型加载成功")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None
    else:
        logger.error(f"模型文件 {model_path} 不存在")
        return None

# 加载模型
model = load_trained_model()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': '模型未加载'}), 500
    
    try:
        data = request.json
        if 'features' not in data:
            return jsonify({'success': False, 'error': '缺少features字段'}), 400
            
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction[0][0])
        })
    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)