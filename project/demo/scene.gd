extends Node

func _ready() -> void:
	print("Hello GDScript!")
	print(OnnxRunner)
	print("HERE")
	var m = OnnxRunner.load_model(r"D:\godot\godot_onnx_extension\project\demo\model.onnx")
	print("THERE")
	m.input_shape(0);
	print("HOO HOO")
	print(m.input_shape(0))
	print("wOO HOO")
	print(m.input_name(0))
	print("OH dear")
	print(m.input_shape(2))
	var out=m.run(PackedFloat32Array([1,2,3]))
	print("GOT")
	print(out)
