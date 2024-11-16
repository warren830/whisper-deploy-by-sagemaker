import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import boto3
from datetime import datetime

def deploy_model():
    role = sagemaker.get_execution_role()
    
    # 生成唯一名称
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    endpoint_name = f"roberta-chinese-classifier-{timestamp}"
    
    # 配置环境变量
    environment = {
        "MODEL_NAME": "thu-coai/roberta-base-cold",
        "HF_TASK": "text-classification",
        "TRANSFORMERS_CACHE": "/tmp/transformers_cache",  # 设置缓存目录
        "HF_TRUST_REMOTE_CODE": "1"  # 信任远程代码
    }
    
    # 创建Hugging Face模型对象
    huggingface_model = HuggingFaceModel(
        model_data=None,
        role=role,
        transformers_version="4.26.0",
        pytorch_version="1.13.1",
        py_version="py39",
        entry_point="inference.py",
        source_dir="code",
        env=environment
    )
    
    # 部署模型
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name
    )
    
    print(f"Endpoint name: {endpoint_name}")
    return predictor, endpoint_name

if __name__ == "__main__":
    predictor, endpoint_name = deploy_model()
    print(f"Model deployed successfully!")