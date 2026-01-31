"""
测试幸福感预测API
"""

import requests
import json

# Flask应用基础URL（根据你的配置调整）
BASE_URL = "http://localhost:5000"

def test_prediction_api():
    """测试预测API"""
    print("开始测试幸福感预测API...")

    # 测试数据
    test_data = {
        "education": 4,      # 高中
        "income": 50000,     # 5万年收入
        "health": 4,         # 一般健康
        "marital_status": 3, # 已婚
        "age": 35,
        "gender": 1,         # 男
        "family_income": 80000,
        "work_status": 1,
        "floor_area": 100
    }

    try:
        # 测试单个预测
        print("\n1. 测试单个预测...")
        response = requests.post(f"{BASE_URL}/api/prediction/predict",
                               json=test_data,
                               timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✓ 单个预测成功:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"✗ 单个预测失败: {response.status_code}")
            print(response.text)

        # 测试批量预测
        print("\n2. 测试批量预测...")
        batch_data = {
            "predictions": [
                test_data,
                {**test_data, "age": 25, "income": 30000},
                {**test_data, "age": 50, "health": 2}
            ]
        }

        response = requests.post(f"{BASE_URL}/api/prediction/batch_predict",
                               json=batch_data,
                               timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✓ 批量预测成功:")
            print(f"  预测数量: {result['data']['total']}")
            for i, pred in enumerate(result['data']['results'][:2]):  # 只显示前2个
                print(f"  样本{i+1}: 幸福感={pred.get('prediction', 'N/A')}, 置信度={pred.get('confidence', 'N/A')}")
        else:
            print(f"✗ 批量预测失败: {response.status_code}")
            print(response.text)

        # 测试获取模型信息
        print("\n3. 测试获取模型信息...")
        response = requests.get(f"{BASE_URL}/api/prediction/model_info", timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✓ 获取模型信息成功:")
            model_info = result['data']
            print(f"  模型名称: {model_info['model_name']}")
            print(f"  R²得分: {model_info['metrics']['r2_score']}")
            print(f"  RMSE: {model_info['metrics']['rmse']}")
            print(f"  特征数量: {len(model_info['feature_columns'])}")
        else:
            print(f"✗ 获取模型信息失败: {response.status_code}")
            print(response.text)

        # 测试健康检查
        print("\n4. 测试健康检查...")
        response = requests.get(f"{BASE_URL}/api/prediction/health", timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✓ 健康检查成功:")
            print(f"  服务状态: {result['data']['status']}")
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            print(response.text)

        # 测试获取示例输入
        print("\n5. 测试获取示例输入...")
        response = requests.get(f"{BASE_URL}/api/prediction/sample_input", timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✓ 获取示例输入成功")
            sample = result['data']['sample_input']
            print(f"  示例教育水平: {sample['education']}")
            print(f"  示例收入: {sample['income']}")
            print(f"  示例年龄: {sample['age']}")
        else:
            print(f"✗ 获取示例输入失败: {response.status_code}")
            print(response.text)

        print("\n🎉 所有API测试完成！")

    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保Flask应用正在运行在 http://localhost:5000")
        print("启动命令: python app.py")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    test_prediction_api()
