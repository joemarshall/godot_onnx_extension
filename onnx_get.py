import subprocess
import tempfile
import shutil
from pathlib import Path

VERSION="1.17.0"

SOURCES={
        "windows":f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-win-x64-{VERSION}.zip",
        "macos":f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-osx-universal2-{VERSION}.tgz",
        "linux":f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-{VERSION}.tgz",
        "linuxbsd":f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-{VERSION}.tgz",
        "android":f"https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/{VERSION}/onnxruntime-android-{VERSION}.aar"
}

def get_onnx_path(env):
    source_url=SOURCES[env['platform']]
    print(source_url.split("/")[-1])
    onnx_path=Path("onnxruntimes", source_url.split("/")[-1]).with_suffix("")
    print("Onnx path=",onnx_path)
    return onnx_path
    
    
def get_onnx(env,target_include_files):
    source_url=SOURCES[env['platform']]
    print(f"Getting onnx runtime from: {source_url}")
    file_name=source_url.split("/")[-1]
    with tempfile.TemporaryDirectory() as td:
        tmp_file=Path(td) / file_name
        print(f"Downloading to{tmp_file} from {source_url}")
        subprocess.check_call(["curl","-L","-o",tmp_file,source_url])
        if tmp_file.suffix==".aar":
            shutil.unpack_archive(tmp_file,extract_dir=td,format="zip")
            onnx_path = Path(get_onnx_path(env))            
            (onnx_path / "include").mkdir(parents=True,exist_ok=True)
            (onnx_path / "lib").mkdir(parents=True,exist_ok=True)
            # android aar file 
            # rename to zip and unpack:
            # 1) jni things (the lib whatever.so) from /jni/platform_name/...
            print("copying onnx aar into project structure")
            arch = env['arch']
            for x in Path(td).glob("jni/{arch}*/libonnxruntime.so"):
                print(f"Copying from AAR: {x}")
                shutil.copy2(x,onnx_path / "lib")
            # 2) headers (from /headers)
            for x in Path(td).glob("headers/*.h"):
                print(f"Copying from AAR: {x}")
                shutil.copy2(x,onnx_path / "include")
            (Path(get_onnx_path(env)) / ".download_time").write_bytes(b"")
        else:
            target_path= Path("onnxruntimes")
            target_path.mkdir(parents=True,exist_ok=True)
            print(f"Unpacking to {target_path}")
            shutil.unpack_archive(tmp_file,extract_dir=target_path)
            (Path(get_onnx_path(env)) / ".download_time").write_bytes(b"")

