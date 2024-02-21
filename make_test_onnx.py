import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, activation="relu"))

input_signature = [tf.TensorSpec([1, 3], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=17)
onnx.save(onnx_model, "project/demo/model.onnx")