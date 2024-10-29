import requests
import json
import numpy as np

url = 'http://1899393106126557.cn-beijing.pai-eas.aliyuncs.com/api/predict/tf_server_simple/v1/models/model:predict'
headers = {'Content-Type': 'application/json','Authorization':'N2IxNWY2ZmFlMTliNDYxZGUwNzZiNDM4ZjY0OTVjODRkNjU2NjE3Zg=='}

# 准备输入数据
# 示例数据：一个形状为 [1, 28, 28, 1] 的张量
input_data =  [[0.5, 0.2], [0.1, 0.8],[0.0, 0.0],[1.0, 1.0],[2.0, 2.0]]

# 将输入数据转换为JSON格式
request_data = json.dumps({
    "inputs": input_data  # 将numpy数组转换为list
})

response = requests.post(url, data=request_data, headers=headers)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
    
    
import json
import base64

# 原始数据
data = {"inputs": [[0.5, 0.2], [0.1, 0.8]]}
data =  {"inputs":[[1, 2,0.1, 0.8],[1, 2,0.2, 0.8],[2, 2,0.2, 0.8],[3, 2,0.2, 0.8],[4, 2,0.2, 0.8]]}
# 将数据转换为JSON字符串
json_str = json.dumps(data)

# 将字符串转换为字节串
json_bytes = json_str.encode('utf-8')

# 对字节串进行Base64编码
encoded_bytes = base64.b64encode(json_bytes)

# 将编码结果转换回字符串
encoded_str = encoded_bytes.decode('utf-8')

print(encoded_str)