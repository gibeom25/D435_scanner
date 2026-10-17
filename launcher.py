"""통합 실행 화면 (런처).

스캔·편집·다듬기·합치기·다시 보정을 버튼 하나로 실행한다. 각 작업은 같은 프로그램을 별도 프로세스로
실행하므로(scanner.py 명령줄 모드), Open3D 창이 혹시 비정상 종료돼도 런처는 계속 살아 있고 결과를 알려준다.
작업 출력은 작업별 로그 파일에 남고, 런처 아래쪽에 실시간으로 보인다.
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import open3d.visualization.gui as gui

APP_NAME = "D435 Scanner"
FROZEN = getattr(sys, "frozen", False)
HERE = os.path.dirname(sys.executable if FROZEN else os.path.abspath(__file__))
CONFIG_PATH = os.path.join(os.environ.get("APPDATA", HERE), "D435Scanner", "config.json")
USER_DIR = os.path.join(os.environ.get("LOCALAPPDATA", HERE), "D435Scanner")


def _writable(d):
    try:
        os.makedirs(d, exist_ok=True)
        test = os.path.join(d, ".write_test")
        with open(test, "w"):
            pass
        os.remove(test)
        return True
    except OSError:
        return False


# 설치 폴더가 쓰기 금지(Program Files 등)면 로그와 기본 저장 폴더를 사용자 폴더로
LOG_DIR = os.path.join(HERE, "logs")
if not _writable(LOG_DIR):
    LOG_DIR = os.path.join(USER_DIR, "logs")
DEFAULT_SCANS = os.path.join(HERE if LOG_DIR.startswith(HERE) else USER_DIR, "scans")
KEEP_LOGS = 30
LOG_LINES = 22


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8-sig") as fp:   # 설치 프로그램이 BOM을 붙여 씀
            return json.load(fp)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)


def _exit_text(code):
    if code == 0:
        return "정상 종료"
    if code < 0 or code > 0xFFFF:
        return f"비정상 종료 (오류 코드 0x{code & 0xFFFFFFFF:08X})"
    return f"실패 (코드 {code})"


def _tail(path, n):
    try:
        with open(path, "rb") as fp:
            fp.seek(0, 2)
            size = fp.tell()
            fp.seek(max(0, size - 16000))
            data = fp.read().decode("utf-8", "replace")
    except OSError:
        return ""
    # 진행률처럼 \r로 덮어쓰는 줄은 마지막 상태만
    lines = [ln.split("\r")[-1] for ln in data.splitlines()]
    return "\n".join(ln[:110] for ln in lines[-n:])


class Task:
    def __init__(self, name, proc, log_path):
        self.name, self.proc, self.log_path = name, proc, log_path
        self.started = time.time()
        self.code = None


class Launcher:
    def __init__(self):
        self.cfg = load_config()
        self.scans_dir = self.cfg.get("scans_dir") or DEFAULT_SCANS
        self.tasks = []
        self.shown = None          # 로그를 보여주는 작업
        self.closing = False

        app = gui.Application.instance
        self.window = app.create_window(APP_NAME, 1180, 760)
        w = self.window
        em = w.theme.font_size
        left = gui.Vert(0.4 * em, gui.Margins(em, em, em, em))
        right = gui.Vert(0.4 * em, gui.Margins(em, em, em, em))
        cols = [left]

        class _Col:   # header/hint/buttons가 지금 채우는 칸에 붙도록
            def add_child(self, x):
                cols[0].add_child(x)
        root = _Col()

        def header(text):
            lb = gui.Label(text)
            lb.text_color = gui.Color(1.0, 0.72, 0.3)
            if cols[0].get_children():
                root.add_child(gui.Label(""))
            root.add_child(lb)

        def hint(text):
            lb = gui.Label(text)
            lb.text_color = gui.Color(0.6, 0.6, 0.6)
            root.add_child(lb)

        def buttons(*items):
            row = gui.Horiz(0.4 * em)
            for text, fn in items:
                b = gui.Button(text)
                b.set_on_clicked(fn)
                row.add_child(b)
            root.add_child(row)

        title = gui.Label("RealSense D435 3D 스캐너")
        root.add_child(title)

        # ---- 카메라
        header("카메라")
        self.cam_label = gui.Label("확인 중...")
        root.add_child(self.cam_label)
        buttons(("카메라 다시 확인", self._check_camera))

        # ---- 스캔
        header("1. 스캔")
        buttons(("물체 스캔", lambda: self._scan(False)), ("사람 스캔 (회전의자)", lambda: self._scan(True)))
        hint("스캔 창: SPACE 시작/일시정지 · R 다시 · ENTER 끝내기 · ESC 취소")
        grid = gui.VGrid(2, 0.4 * em)
        self.voxel = gui.Slider(gui.Slider.DOUBLE)
        self.voxel.set_limits(1.0, 5.0)
        self.voxel.double_value = float(self.cfg.get("refine_voxel_mm", 2.0))
        grid.add_child(gui.Label("정밀 재구성 해상도 (mm)"))
        grid.add_child(self.voxel)
        self.preset = gui.Combobox()
        for p in ("high_accuracy", "default", "high_density"):
            self.preset.add_item(p)
        self.preset.selected_text = self.cfg.get("preset", "high_accuracy")
        grid.add_child(gui.Label("깊이 프리셋"))
        grid.add_child(self.preset)
        root.add_child(grid)
        self.cb_norefine = gui.Checkbox("정밀 재구성 건너뛰기 (빠르지만 거침)")
        root.add_child(self.cb_norefine)
        hint("해상도: 작을수록 세밀하지만 느림 · 머리카락에 구멍이 많으면 프리셋 default")

        # ---- 열기 / 편집
        header("2. 열기 · 편집")
        buttons(("스캔 열기 (편집기)", self._open_editor), ("다듬기 도구", self._open_repair))
        hint("편집기: 박스로 자르기 후 프린팅용 STL 저장 · 다듬기: 조각 잇기, 구멍 메우기, 스무딩")

        # ---- 후처리
        header("3. 합치기 · 다시 보정")
        buttons(("최근 두 스캔 합치기", lambda: self._run("스캔 합치기", ["--merge", "latest2"])),
                ("두 스캔 골라 합치기", self._merge_pick))
        buttons(("스캔 하나 다시 보정", self._recorrect_pick),
                ("모든 스캔 다시 보정", lambda: self._run("모두 다시 보정", ["--recorrect"])))
        hint("녹화(frames.npz)가 있는 스캔만 가능 · 이전 결과는 scene_prev.ply로 보관")

        # ---- 오른쪽 칸: 저장 폴더 · 진행 상황
        cols[0] = right
        header("저장 폴더")
        self.dir_label = gui.Label(self.scans_dir)
        root.add_child(self.dir_label)
        buttons(("폴더 바꾸기", self._change_dir), ("폴더 열기", lambda: self._explore(self.scans_dir)))

        # ---- 진행 상황
        header("진행 상황")
        self.status = gui.Label("대기 중")
        root.add_child(self.status)
        self.log = gui.Label("\n" * (LOG_LINES - 1))
        self.log.text_color = gui.Color(0.75, 0.75, 0.75)
        root.add_child(self.log)
        buttons(("작업 중지", self._stop), ("로그 폴더 열기", lambda: self._explore(LOG_DIR)))

        body = gui.Horiz(0)
        body.add_child(left)
        body.add_child(right)
        w.add_child(body)
        w.set_on_close(self._on_close)

        threading.Thread(target=self._poll_loop, daemon=True).start()
        self._check_camera()

    # ------------------------------------------------------------ 작업 실행

    def _command(self, args):
        if FROZEN:
            return [sys.executable] + args
        return [sys.executable, os.path.join(HERE, "main.py")] + args

    def _run(self, name, args):
        os.makedirs(self.scans_dir, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"{stamp}_{name.replace(' ', '_')}.log")
        env = dict(os.environ, RS_SCANS_DIR=self.scans_dir, RS_LOG_PATH=log_path,
                   PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        try:
            proc = subprocess.Popen(self._command(args), cwd=HERE, env=env,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError as e:
            self.status.text = f"실행 실패: {e}"
            return
        t = Task(name, proc, log_path)
        self.tasks.append(t)
        self.shown = t
        self.status.text = f"실행 중: {name}"
        self.log.text = "\n" * (LOG_LINES - 1)
        self._prune_logs()

    def _scan(self, subject):
        self.cfg["refine_voxel_mm"] = round(self.voxel.double_value, 2)
        self.cfg["preset"] = self.preset.selected_text
        save_config(self.cfg)
        args = ["--refine-voxel", f"{self.voxel.double_value / 1000:.4f}", "--preset", self.preset.selected_text]
        if subject:
            args.append("--subject")
        if self.cb_norefine.checked:
            args.append("--no-refine")
        if any(t.code is None and t.name.endswith("스캔") for t in self.tasks):
            self.status.text = "이미 스캔 중입니다. 카메라는 한 번에 하나의 스캔만 쓸 수 있습니다."
            return
        self._run("사람 스캔" if subject else "물체 스캔", args)

    def _open_editor(self):
        self._run("편집기", ["--load"])

    def _open_repair(self):
        self._pick_file("다듬을 메시 고르기", [(".ply .obj .stl", "메시")],
                        lambda p: self._run("다듬기", ["--repair", p]))

    def _merge_pick(self):
        def second(a):
            self._pick_dir("두 번째 스캔 폴더 고르기",
                           lambda b: self._run("스캔 합치기", ["--merge", a, b]))
        self._pick_dir("첫 번째 스캔 폴더 고르기 (예: 정면)", second)

    def _recorrect_pick(self):
        self._pick_dir("다시 보정할 스캔 폴더 고르기", lambda p: self._run("다시 보정", ["--recorrect", p]))

    def _stop(self):
        t = self.shown
        if t is None or t.code is not None:
            self.status.text = "실행 중인 작업이 없습니다."
            return
        t.proc.terminate()
        self.status.text = f"중지함: {t.name}"

    # ------------------------------------------------------------ 상태 갱신 (백그라운드에서 0.5초마다)

    def _poll_loop(self):
        while not self.closing:
            time.sleep(0.5)
            for t in self.tasks:
                if t.code is None:
                    c = t.proc.poll()
                    if c is not None:
                        t.code = c
            t = self.shown
            if t is None:
                continue
            text = _tail(t.log_path, LOG_LINES)
            running = sum(1 for x in self.tasks if x.code is None)
            if t.code is None:
                status = f"실행 중: {t.name} ({int(time.time() - t.started)}초)"
            else:
                status = f"{t.name}: {_exit_text(t.code)}"
            if running > (t.code is None):
                status += f"  · 다른 작업 {running - (t.code is None)}개 실행 중"
            gui.Application.instance.post_to_main_thread(self.window, lambda s=status, x=text: self._show(s, x))

    def _show(self, status, text):
        if self.closing:
            return
        self.status.text = status
        lines = text.splitlines()
        self.log.text = "\n".join(lines + [""] * (LOG_LINES - len(lines)))

    # ------------------------------------------------------------ 카메라

    def _check_camera(self):
        self.cam_label.text = "확인 중..."

        def work():
            try:
                import pyrealsense2 as rs
                devs = rs.context().query_devices()
                if len(devs) == 0:
                    msg = "카메라 없음: D435를 USB 3 포트(파란색)에 꽂고 [카메라 다시 확인]"
                else:
                    d = devs[0]
                    info = lambda k: d.get_info(k) if d.supports(k) else "?"
                    usb = info(rs.camera_info.usb_type_descriptor)
                    msg = (f"연결됨: {info(rs.camera_info.name)} (펌웨어 {info(rs.camera_info.firmware_version)},"
                           f" USB {usb})")
                    if usb.startswith("2"):
                        msg += "\nUSB 2로 연결됨: USB 3 포트·케이블이면 훨씬 안정적입니다"
            except Exception as e:  # noqa: BLE001
                msg = f"카메라 확인 실패: {e}"
            gui.Application.instance.post_to_main_thread(self.window, lambda: self._set_cam(msg))
        threading.Thread(target=work, daemon=True).start()

    def _set_cam(self, msg):
        self.cam_label.text = msg
        self.window.set_needs_layout()   # 줄 수가 바뀌면 아래 버튼과 겹치지 않게 다시 배치

    # ------------------------------------------------------------ 폴더·파일

    def _change_dir(self):
        def done(p):
            self.scans_dir = p
            self.dir_label.text = p
            self.cfg["scans_dir"] = p
            save_config(self.cfg)
        self._pick_dir("스캔 저장 폴더 고르기", done)

    def _pick_dir(self, title, on_done):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, title, self.window.theme)
        dlg.set_path(self.scans_dir if os.path.isdir(self.scans_dir) else HERE)
        self._show_dialog(dlg, on_done)

    def _pick_file(self, title, filters, on_done):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, title, self.window.theme)
        for ext, desc in filters:
            dlg.add_filter(ext, desc)
        dlg.set_path(self.scans_dir if os.path.isdir(self.scans_dir) else HERE)
        self._show_dialog(dlg, on_done)

    def _show_dialog(self, dlg, on_done):
        dlg.set_on_cancel(self.window.close_dialog)

        def done(p):
            self.window.close_dialog()
            # 대화상자 콜백 안에서 바로 다음 대화상자를 띄우면 불안정 → 콜백이 끝난 뒤 처리
            gui.Application.instance.post_to_main_thread(self.window, lambda: on_done(os.path.normpath(p)))
        dlg.set_on_done(done)
        self.window.show_dialog(dlg)

    @staticmethod
    def _explore(path):
        os.makedirs(path, exist_ok=True)
        os.startfile(path)

    def _prune_logs(self):
        try:
            logs = sorted(f for f in os.listdir(LOG_DIR) if f.endswith(".log") and f != "last_run.log")
            for f in logs[:-KEEP_LOGS]:
                os.remove(os.path.join(LOG_DIR, f))
        except OSError:
            pass

    def _on_close(self):
        self.closing = True
        return True   # 작업 창(스캔·편집기)은 따로 떠 있으므로 그대로 둔다


def run_launcher():
    from editor import init_app
    app = init_app()
    Launcher()
    app.run()
