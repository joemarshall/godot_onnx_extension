# Godot Onnx Extension

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
