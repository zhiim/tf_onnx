import onnxruntime as onnxrt
import numpy as np

class_names = ['CSNJ', 'CW', 'LFM', 'MTJ', 'PBNJ', 'PPNJ']

with open("input/STFT.csv") as file_name:
    test_img = np.loadtxt(file_name, delimiter=",")

test_img = np.array(test_img).astype(np.float32)
test_img = np.expand_dims(test_img, axis=2)
test_img = np.expand_dims(test_img, 0)  # 将三维输入图像拓展成四维张量

# load model
model = onnxrt.InferenceSession("model/ResNet.onnx")
print("loading model success!")


# input
sf_input = {model.get_inputs()[0].name:test_img}

# output
output = model.run(None, sf_input)
print("get output of spatial filter success!")
output = np.array(output)
pred = class_names[output.argmax()]
print(pred)
with open("output/predict.txt", 'w', encoding='utf-8') as f:
    f.write(pred)
    f.close()
