import subprocess
import tempfile
import shutil
from pathlib import Path

VERSION = "1.17.0"

SOURCES = {
    "windows": f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-win-x64-{VERSION}.zip",
    "macos": f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-osx-universal2-{VERSION}.tgz",
    "linux": f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-{VERSION}.tgz",
    "linuxbsd": f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-{VERSION}.tgz",
    "android": f"https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/{VERSION}/onnxruntime-android-{VERSION}.aar",
}


def get_onnx_path(env):
    source_url = SOURCES[env["platform"]]
    print(source_url.split("/")[-1])
    onnx_src_path = Path("onnxruntimes", source_url.split("/")[-1]).with_suffix("")
    onnx_dest_path = Path("bin") / Path(env["platform"]) / (env["arch"])
    print("Onnx path=", onnx_src_path, onnx_dest_path)
    return onnx_src_path, onnx_dest_path


def get_onnx(env, install_folder):
    print(f"INSTALL_FOLDER:{install_folder}")
    source_url = SOURCES[env["platform"]]
    print(f"Getting onnx runtime from: {source_url}")
    file_name = source_url.split("/")[-1]
    with tempfile.TemporaryDirectory() as td:
        tmp_file = Path(td) / file_name
        print(f"Downloading to{tmp_file} from {source_url}")
        subprocess.check_call(["curl", "-L", "-o", tmp_file, source_url])
        src_path, inst_path = get_onnx_path(env)
        onnx_path = Path(src_path)
        onnx_install_path = Path(install_folder) / inst_path
        (onnx_path / "include").mkdir(parents=True, exist_ok=True)
        (onnx_path / "lib").mkdir(parents=True, exist_ok=True)
        onnx_install_path.mkdir(parents=True, exist_ok=True)

        if tmp_file.suffix == ".aar":
            shutil.unpack_archive(tmp_file, extract_dir=td, format="zip")
            # android aar file
            # rename to zip and unpack:
            # 1) jni things (the lib whatever.so) from /jni/platform_name/...
            print("copying onnx aar into project structure")
            arch = env["arch"]
            for x in Path(td).glob(f"jni/{arch}*/libonnxruntime.so"):
                print(f"Copying lib from AAR to build folder: {x}")
                shutil.copy2(x, onnx_path / "lib")
            for x in Path(td).glob(f"jni/{arch}*/libonnxruntime.so"):
                print(
                    f"Copying lib from AAR to install folder {onnx_install_path}: {x}"
                )
                shutil.copy2(x, onnx_install_path)

            # 2) headers (from /headers)
            for x in Path(td).glob("headers/*.h"):
                print(f"Copying include from AAR: {x}")
                shutil.copy2(x, onnx_path / "include")
            (Path(get_onnx_path(env)[0]) / ".download_time").write_bytes(b"")
        else:
            target_path = onnx_path.parent
            print(f"Unpacking to {target_path}")
            shutil.unpack_archive(tmp_file, extract_dir=target_path)

            for x in (onnx_path / "lib").glob(f"*{env['SHLIBSUFFIX']}"):
                print(f"Copying lib to install folder {onnx_install_path}: {x}")
                shutil.copy2(x, onnx_install_path)

            (Path(get_onnx_path(env)[0]) / ".download_time").write_bytes(b"")
