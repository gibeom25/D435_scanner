# PyInstaller 빌드 설정:  packaging\build.bat 로 실행
# 결과: dist\D435 Scanner\D435 Scanner.exe (+ _internal 폴더). 설치 파일은 installer.iss가 이것을 묶는다.
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = collect_data_files("open3d") + collect_data_files("trimesh")
binaries = collect_dynamic_libs("open3d") + collect_dynamic_libs("pyrealsense2")

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "open3d.cpu.pybind",
        "pyrealsense2.pyrealsense2",
        "scanner", "launcher", "editor", "repair", "sculpt", "scan", "refine", "mesh_tools", "gui_safety",
        "pymeshfix", "shapely", "scipy.sparse.csgraph", "scipy.spatial",
    ],
    # 필요 없는 큰 패키지 제외. dash·plotly는 Open3D가 시작할 때 무조건 불러오므로 빼면 안 됨.
    excludes=["matplotlib", "IPython", "ipywidgets", "jupyter", "notebook",
              "pandas", "tkinter", "PyQt5", "PySide6", "pytest"],
    noarchive=False,
)
# VC++ 런타임: 다른 패키지에 들어 있던 오래된 MSVCP140(14.16)이 잡히면 Open3D가
# "DLL 초기화 루틴을 실행할 수 없습니다"로 시작조차 못 한다 → 시스템의 최신 버전으로 바꿔 넣는다.
# (앱 폴더에 함께 넣으므로 설치하는 PC에 VC++ 재배포 패키지가 없어도 동작)
VCRT = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"]
SYS32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
a.binaries = [b for b in a.binaries if os.path.basename(b[0]).lower() not in VCRT]
for name in VCRT:
    src = os.path.join(SYS32, name)
    if os.path.exists(src):
        a.binaries.append((name, src, "BINARY"))

pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="D435 Scanner",
    icon=os.path.join(SPECPATH, "icon.ico"),
    console=False,          # 창 프로그램 (작업 출력은 런처 화면과 logs 폴더에)
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="D435 Scanner")
