#!/usr/bin/env python
from glob import glob
from pathlib import Path
import onnx_get
import os
import re
import godotconfig

scons_cache_path = os.environ.get("SCONS_CACHE")
cacher=None
if scons_cache_path != None:
    cacher=CacheDir(scons_cache_path)
    print("Scons cache enabled... (path: '" + scons_cache_path + "')")


# TODO: Do not copy environment after godot-cpp/test is updated <https://github.com/godotengine/godot-cpp/blob/master/test/SConstruct>.
env = SConscript("godot-cpp/SConstruct")
onnx_src_path, onnx_dest_path = onnx_get.get_onnx_path(env)


# Add source files.
env.Append(CPPPATH=["src/", f"{onnx_src_path}/include"])
sources = Glob("src/*.cpp")
env.Append(LIBPATH=f"{onnx_src_path}/lib/")
env.Append(LIBS=["onnxruntime"])

# Find gdextension path even if the directory or extension is renamed (e.g. project/addons/example/example.gdextension).
(extension_path,) = glob("project/addons/*/*.gdextension")

# Find the addon path (e.g. project/addons/example).
addon_path = Path(extension_path).parent
onnx_dest_path = addon_path / onnx_dest_path

# Find the project name from the gdextension file (e.g. example).
project_name = Path(extension_path).stem


# download onnx and copy to destination path
def download_onnx_release(target, source, env):
    global addon_path
    onnx_get.get_onnx(env, addon_path)


get_onnx_cmd = env.Command(
    f"{onnx_src_path}/.download_time",
    source="onnx_get.py",
    action=download_onnx_release,
)
# make sure onnx is got before we try to build the source that uses it
Depends("src/onnx_mingw_overrides.h", get_onnx_cmd)

# Create the library target (e.g. libexample.linux.debug.x86_64.so).
debug_or_release = "release" if env["target"] == "template_release" else "debug"


if env["platform"] == "macos":
    target_lib_name = "{0}/bin/lib{1}.{2}.{3}.framework/{1}.{2}.{3}".format(
        addon_path,
        project_name,
        env["platform"],
        debug_or_release,
    )
else:
    # shared library has to be in same path as onnx libraries,
    # otherwise we have to deal with library search paths and
    # it could be a pain
    target_lib_name = "{}/lib{}.{}.{}.{}{}".format(
        onnx_dest_path,
        project_name,
        env["platform"],
        debug_or_release,
        env["arch"],
        env["SHLIBSUFFIX"],
    )

library = env.SharedLibrary(
    target=target_lib_name,
    source=sources,
)

# make gdextension part for this build setting
# they get merged in github actions


def make_gdextension_part(target, source, env):
    output_path = Path(str(target[0])).absolute()
    target_lib = Path(env["target_lib"]).absolute()
    extension_base = Path(env["addon_path"])
    # make library key
    debug_or_release = env["debug_or_release"]
    platform_key = f"{env['platform']}.{debug_or_release}.{env['arch']}"
    lib_value = target_lib.relative_to(output_path.parent)
    extension_values = []
    extension_values.append(("libraries", {platform_key: lib_value}))
    deps_keys = {}
    for lib_name in target_lib.parent.glob(f"*{env['SHLIBSUFFIX']}"):
        if lib_name != target_lib:
            deps_keys[lib_name.relative_to(output_path.parent)] = ""
    extension_values.append(("dependencies", {platform_key: deps_keys}))
    output_path.write_text(godotconfig.get_as_text(extension_values))


env["addon_path"] = addon_path
env["debug_or_release"] = debug_or_release
env["target_lib"] = target_lib_name
make_gdextension_cmd = env.Command(
    f"{extension_path}.{env['platform']}.{env['arch']}.{env['debug_or_release']}.part",
    action=make_gdextension_part,
    source=None,
    addon_path=addon_path,
    debug_or_release=debug_or_release,
    target_lib_name=target_lib_name,
)
Depends(library, get_onnx_cmd)
Depends(make_gdextension_cmd, library)


def check_cache_hit(target,source,env):
    print("Checking cache")
    cd = env.get_CacheDir()
    if cd!=None:
        print(f"Cache hit ratio:{cd.hit_ratio}, missed {cd.misses}")
        if cd.hit_ratio<60.0 and cd.misses>5:
            Path(str(target[0])).write_text("cache-hit=false")
        else:
            Path(str(target[0])).write_text("cache-hit=true")

# remove .cache_used from cache or else the command will be skipped
cd=env.get_CacheDir()
if cd:
    _,cache_path=cd.cachepath(env.File(".cache_used"))
    Path(cache_path).unlink(missing_ok=True)

cache_check_command=env.Command(".cache_used",source=[],action=check_cache_hit)
env.Depends(cache_check_command,make_gdextension_cmd)
env.AlwaysBuild(cache_check_command)
env.AddPostAction(make_gdextension_cmd,cache_check_command)
Default(cache_check_command)
