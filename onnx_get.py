import subprocess
import tempfile
import shutil
from pathlib import Path

VERSION="1.17.0"

SOURCES={
        "windows":f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-win-x64-{VERSION}.zip",
        "macos":f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-osx-universal2-{VERSION}.tgz",
        "linuxbsd":f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-{VERSION}.tgz",
        "android":f"https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-aarch64-{VERSION}.tgz"
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
        tmp_file=Path(td,file_name)
        print(f"Downloading to{tmp_file}")
        subprocess.check_call(["curl","-L","-o",tmp_file,source_url])
        target_path= Path("onnxruntimes")
        target_path.mkdir(parents=True,exist_ok=True)
        print(f"Unpacking to {target_path}")
        shutil.unpack_archive(tmp_file,extract_dir=target_path)
