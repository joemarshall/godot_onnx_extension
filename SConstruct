#!/usr/bin/env python
from glob import glob
from pathlib import Path
import onnx_get
import os


scons_cache_path = os.environ.get("SCONS_CACHE")
if scons_cache_path != None:
     CacheDir(scons_cache_path)
     print("Scons cache enabled... (path: '" + scons_cache_path + "')")

def download_onnx_release(target,source,env):
    onnx_get.get_onnx(env,target)

# TODO: Do not copy environment after godot-cpp/test is updated <https://github.com/godotengine/godot-cpp/blob/master/test/SConstruct>.
env = SConscript("godot-cpp/SConstruct")
onnx_path=onnx_get.get_onnx_path(env)

get_onnx_cmd = env.Command(f'{onnx_path}/.download_time',source='onnx_get.py',action=download_onnx_release)
# Add source files.
env.Append(CPPPATH=["src/",f"{onnx_path}/include"])
env.Append(CCFLAGS="-fexceptions")
sources = Glob("src/*.cpp")
env.Append(LIBPATH=f"{onnx_path}/lib/")
env.Append(LIBS=["onnxruntime"])


# Find gdextension path even if the directory or extension is renamed (e.g. project/addons/example/example.gdextension).
(extension_path,) = glob("project/addons/*/*.gdextension")

# Find the addon path (e.g. project/addons/example).
addon_path = Path(extension_path).parent

# Find the project name from the gdextension file (e.g. example).
project_name = Path(extension_path).stem


# Create the library target (e.g. libexample.linux.debug.x86_64.so).
debug_or_release = "release" if env["target"] == "template_release" else "debug"
if env["platform"] == "macos":
    library = env.SharedLibrary(
        "{0}/bin/lib{1}.{2}.{3}.framework/{1}.{2}.{3}".format(
            addon_path,
            project_name,
            env["platform"],
            debug_or_release,
        ),
        source=sources,
    )
else:
    library = env.SharedLibrary(
        "{}/bin/lib{}.{}.{}.{}{}".format(
            addon_path,
            project_name,
            env["platform"],
            debug_or_release,
            env["arch"],
            env["SHLIBSUFFIX"],
        ),
        source=sources,
    )
Depends(library,get_onnx_cmd)

Default(library)
