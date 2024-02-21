# Godot Onnx Extension

Extension to allow running inference on [onnx](https://onnxruntime.ai/) models within Godot.

Usage:
1) Load a model with `OnnxRunner.load_model` pointing at either an absolute path, or a godot `res://` project relative path.
2) run inference on a model using 'model.run(PackedFloat32Array)`. Bear in mind that in godot, a PackedFloat32Array is one dimensional, so you need to flatten your input data.
   
```
	var m = OnnxRunner.load_model(r"res://demo/model.onnx")
	print(m.num_inputs(0)) 
	print(m.input_shape(0)) 
	print(m.input_name(0))
	print(m.num_outputs(0)) 
	print(m.output_shape(0)) 
	print(m.output_name(0))
	var out=m.run(PackedFloat32Array([1,2,3]))
```


### Repository structure:
- `project/` - Godot project boilerplate.
  - `addons/onnx/` - Files to be distributed to other projects.¹
  - `demo/` - Scenes and scripts for testing and demonstration
- `src/` - Source code of this extension.
- `godot-cpp/` - Submodule needed for GDExtension compilation.

# Building

Build using `scons` as per building godot. `template_debug` and `template_release` targets.
e.g.
for Windows:
`scons target=template_debug platform=windows arch=x86_64`
for Android:
`scons platform=windows arch=arm64`

Build dependencies are the same as for Godot, i.e. scons, C++ build tools, Android NDK / JDK for Android builds etc. Check out
`.github/workflows/build.yml` for working build scripts (or use github actions if you can't be bothered to install all the build dependencies).
