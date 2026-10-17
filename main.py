"""D435 Scanner 진입점.

인자 없이 실행: 통합 실행 화면(런처)
인자와 함께 실행: 명령줄 모드 (python main.py --subject, --load, --repair 경로 ... = scanner.py와 같음)
"""
import multiprocessing
import os
import sys


def _report_startup_error():
    """창 프로그램(exe)은 콘솔이 없으므로, 시작 중 오류는 파일에 남기고 알림 창으로 보여준다."""
    import traceback
    here = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
    text = traceback.format_exc()
    path = "(저장 실패)"
    # 설치 폴더가 쓰기 금지면 사용자 폴더에
    for d in (os.path.join(here, "logs"), os.path.join(os.environ.get("LOCALAPPDATA", here), "D435Scanner", "logs")):
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "startup_error.log"), "w", encoding="utf-8") as fp:
                fp.write(text)
            path = os.path.join(d, "startup_error.log")
            break
        except OSError:
            continue
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, f"{text[-1500:]}\n\n자세한 내용: {path}", "D435 Scanner 시작 오류", 0x10)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        from scanner import run_cli
        sys.exit(run_cli())
    try:
        from launcher import run_launcher
        run_launcher()
    except Exception:  # noqa: BLE001
        _report_startup_error()
        sys.exit(1)
