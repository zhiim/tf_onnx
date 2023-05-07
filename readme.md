Demo for convert tensorflow .h5 model to .onnx model.  
There is a sample script for loading .h5 model and saved it to SavedModel in folder `model_convert`.  
Then, we can install [tf2onnx](https://github.com/onnx/tensorflow-onnx). And run `python -m tf2onnx.convert --saved-model saved_model --output model.onnx` under `model_convert`.  
We can get a output file named `model.onnx`.  
Moving `model.onnx` to `../model`, we can run `predict.py` to get output using `model.onnx`.
