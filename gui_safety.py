"""화면 콜백 안전장치.

Open3D 화면의 버튼·슬라이더·마우스 처리(콜백) 안에서 파이썬 오류가 나면, Open3D는 그 오류를
처리하지 못하고 프로그램 전체가 꺼진다(Windows 오류 0xc000041d). 모든 콜백을 감싸서
오류가 나도 프로그램은 계속 돌고, 오류 내용(어느 동작·어느 줄)은 콘솔과 logs/last_run.log에 남긴다.
"""
import functools
import sys
import traceback

import open3d.visualization.gui as gui

_installed = False
on_error = None   # 선택: 오류 메시지를 화면에 보여줄 함수 (msg) -> None


def _report(where, exc):
    msg = f"[화면 오류] {where}: {type(exc).__name__}: {exc}"
    print(msg, file=sys.stderr)
    traceback.print_exc()
    if on_error is not None:
        try:
            on_error(msg)
        except Exception:  # noqa: BLE001 - 안전장치 안에서 다시 죽지 않게
            pass


def safe(fn, where="", default=None):
    if fn is None or getattr(fn, "_safe_wrapped", False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - 화면 콜백에서는 어떤 오류도 밖으로 내보내지 않음
            _report(where or getattr(fn, "__qualname__", "콜백"), e)
            return default() if callable(default) else default
    wrapper._safe_wrapped = True
    return wrapper


def _patch(cls, method, default=None):
    orig = getattr(cls, method)

    def patched(self, fn, *rest):
        return orig(self, safe(fn, f"{cls.__name__}.{method}", default), *rest)
    setattr(cls, method, patched)


def install():
    """앱 시작 시 한 번 호출."""
    global _installed
    if _installed:
        return
    _installed = True
    for cls, method in ((gui.Button, "set_on_clicked"),
                        (gui.Slider, "set_on_value_changed"),
                        (gui.Checkbox, "set_on_checked"),
                        (gui.ListView, "set_on_selection_changed"),
                        (gui.NumberEdit, "set_on_value_changed"),
                        (gui.Window, "set_on_layout"),
                        (gui.FileDialog, "set_on_done"),
                        (gui.FileDialog, "set_on_cancel")):
        if hasattr(cls, method):
            _patch(cls, method)
    # 마우스 콜백은 결과값이 필요하므로 오류 시 '무시'를 돌려줌
    _patch(gui.SceneWidget, "set_on_mouse", default=lambda: gui.Widget.EventCallbackResult.IGNORED)
    # 백그라운드 작업이 끝나고 화면에 반영하는 함수도 감싼다
    orig_post = gui.Application.post_to_main_thread

    def post(self, window, fn):
        return orig_post(self, window, safe(fn, "작업 결과 반영"))
    gui.Application.post_to_main_thread = post
