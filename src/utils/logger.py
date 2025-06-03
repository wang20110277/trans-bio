import logging
import os

def setup_logger(config):
    """设置日志系统"""
    try:
        # 确保日志目录存在
        log_dir = os.path.dirname(config["logging"]["file"])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 配置日志格式和处理器
        logging.basicConfig(
            level=getattr(logging, config["logging"]["level"], logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config["logging"]["file"]),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    except Exception as e:
        # 如果配置日志失败，使用默认配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.warning(f"日志配置失败，使用默认配置: {e}")
        return logger