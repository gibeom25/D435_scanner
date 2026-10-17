"""스캔 결과에서 박스로 원하는 부분을 골라 3D 프린팅용 STL로 내보내는 편집기."""
import os
import threading

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import mesh_tools as mt

KOREAN_FONT = r"C:\Windows\Fonts\malgun.ttf"


MESH_EXTS = (".ply", ".obj", ".stl")


class Editor:
    def __init__(self, vertices=None, faces=None, colors=None, title="3D 스캔 편집기", default_dir=".",
                 name="", scans_dir=None, open_on_start=True):
        self.default_dir = default_dir
        self.scans_dir = scans_dir or default_dir
        self.title = title
        self.v = self.f = self.c = None
        self.floor_found = False
        self.result = None
        self.show_result = False
        self.busy = False
        self.lo, self.hi = np.zeros(3), np.full(3, 0.3)
        self.box = mt.Box(0, 0.3, 0, 0.3, 0, 0.3)

        app = gui.Application.instance
        self.window = app.create_window(title, 1400, 900)
        w = self.window
        em = w.theme.font_size

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.12, 0.12, 0.14, 1.0])
        self.scene.scene.show_axes(False)

        self.mat_in = rendering.MaterialRecord()
        self.mat_in.shader = "defaultLit"
        self.mat_out = rendering.MaterialRecord()
        self.mat_out.shader = "defaultLitTransparency"
        self.mat_out.base_color = [0.55, 0.55, 0.6, 0.25]
        self.mat_line = rendering.MaterialRecord()
        self.mat_line.shader = "unlitLine"
        self.mat_line.line_width = 2.5
        self.mat_result = rendering.MaterialRecord()
        self.mat_result.shader = "defaultLit"
        self.mat_result.base_color = [0.85, 0.75, 0.55, 1.0]

        # ---------------- 오른쪽 패널
        panel = gui.Vert(0.4 * em, gui.Margins(em, em, em, em))
        b = gui.Button("파일 열기 (메시 / 녹화 frames.npz)")
        b.set_on_clicked(self._open_dialog)
        panel.add_child(b)
        self.file_label = gui.Label("열린 파일 없음")
        self.file_label.text_color = gui.Color(0.7, 0.7, 0.7)
        panel.add_child(self.file_label)
        row_rc = gui.Horiz(0.25 * em)
        b = gui.Button("이 스캔 다시 보정")
        b.set_on_clicked(self._recorrect)
        row_rc.add_child(b)
        row_rc.add_child(gui.Label("처음 장면 N"))
        self.seam_ref = gui.NumberEdit(gui.NumberEdit.INT)
        self.seam_ref.set_limits(1, 40)
        self.seam_ref.int_value = 10
        row_rc.add_child(self.seam_ref)
        panel.add_child(row_rc)
        panel.add_fixed(0.5 * em)

        panel.add_child(gui.Label("1. 주황색 박스로 원하는 부분을 감싸세요"))
        self.floor_warn = gui.Label("")
        self.floor_warn.text_color = gui.Color(1.0, 0.7, 0.3)
        panel.add_child(self.floor_warn)

        self.sliders = {}
        names = [("xmin", "X 최소 (mm)", 0), ("xmax", "X 최대 (mm)", 0), ("ymin", "Y 최소 (mm)", 1),
                 ("ymax", "Y 최대 (mm)", 1), ("zmin", "Z 최소 (바닥)", 2), ("zmax", "Z 최대 (mm)", 2)]
        grid = gui.VGrid(2, 0.25 * em)
        for key, label, axis in names:
            s = gui.Slider(gui.Slider.DOUBLE)
            s.set_limits(1000 * float(self.lo[axis]), 1000 * float(self.hi[axis]))
            s.double_value = 1000 * float(getattr(self.box, key))
            s.set_on_value_changed(lambda val, k=key: self._on_box(k, val))
            grid.add_child(gui.Label(label))
            grid.add_child(s)
            self.sliders[key] = s
        yaw = gui.Slider(gui.Slider.DOUBLE)
        yaw.set_limits(-90.0, 90.0)
        yaw.double_value = 0.0
        yaw.set_on_value_changed(lambda val: self._on_box("yaw_deg", val))
        grid.add_child(gui.Label("회전 (도)"))
        grid.add_child(yaw)
        self.sliders["yaw_deg"] = yaw
        panel.add_child(grid)

        row = gui.Horiz(0.25 * em)
        b = gui.Button("박스 초기화")
        b.set_on_clicked(self._reset_box)
        row.add_child(b)
        b = gui.Button("시점 맞추기")
        b.set_on_clicked(self._fit_camera)
        row.add_child(b)
        panel.add_child(row)

        self.dim_label = gui.Label("")
        panel.add_child(self.dim_label)

        panel.add_fixed(em)
        panel.add_child(gui.Label("2. 프린팅용 처리 옵션"))
        self.cb_largest = gui.Checkbox("가장 큰 덩어리만 남기기")
        self.cb_largest.checked = True
        panel.add_child(self.cb_largest)
        self.cb_watertight = gui.Checkbox("구멍 메워서 닫힌 메시로 (STL 필수 권장)")
        self.cb_watertight.checked = True
        panel.add_child(self.cb_watertight)

        grid2 = gui.VGrid(2, 0.25 * em)
        self.smooth = gui.Slider(gui.Slider.INT)
        self.smooth.set_limits(0, 30)
        self.smooth.int_value = 5
        grid2.add_child(gui.Label("부드럽게"))
        grid2.add_child(self.smooth)
        self.tri = gui.NumberEdit(gui.NumberEdit.INT)
        self.tri.set_limits(0, 5_000_000)
        self.tri.int_value = 200_000
        grid2.add_child(gui.Label("최대 삼각형 수"))
        grid2.add_child(self.tri)
        panel.add_child(grid2)
        hint = gui.Label("(0 = 줄이지 않음)")
        hint.text_color = gui.Color(0.6, 0.6, 0.6)
        panel.add_child(hint)

        panel.add_fixed(em)
        panel.add_child(gui.Label("3. 처리 후 저장"))
        self.btn_process = gui.Button("처리 미리보기")
        self.btn_process.set_on_clicked(self._process)
        panel.add_child(self.btn_process)
        self.cb_show_result = gui.Checkbox("처리 결과만 보기")
        self.cb_show_result.set_on_checked(self._toggle_result)
        self.cb_show_result.enabled = False
        panel.add_child(self.cb_show_result)
        self.btn_save = gui.Button("저장 (STL / OBJ / PLY)")
        self.btn_save.set_on_clicked(self._save_dialog)
        panel.add_child(self.btn_save)
        b = gui.Button("원본 장면 저장 (PLY)")
        b.set_on_clicked(self._save_scene_dialog)
        panel.add_child(b)
        b = gui.Button("다듬기: 조각 잇기 · 구멍 메우기")
        b.set_on_clicked(self._open_repair)
        panel.add_child(b)

        panel.add_fixed(em)
        self.status = gui.Label("[파일 열기]로 스캔을 불러오세요.")
        panel.add_child(self.status)
        self.log_label = gui.Label("")
        self.log_label.text_color = gui.Color(0.7, 0.7, 0.7)
        panel.add_child(self.log_label)

        self.panel = panel
        w.add_child(self.scene)
        w.add_child(panel)
        w.set_on_layout(self._on_layout)

        import gui_safety
        gui_safety.on_error = lambda msg: setattr(self.status, "text", msg + " (프로그램은 계속 동작)")
        if vertices is not None:
            self.load_mesh(vertices, faces, colors, name)
        elif open_on_start:
            # 창이 뜬 직후 파일 선택 창을 띄움
            gui.Application.instance.post_to_main_thread(self.window, self._open_dialog)

    # ------------------------------------------------------------ 불러오기

    def load_mesh(self, vertices, faces, colors, name=""):
        """장면을 새 메시로 교체. vertices 단위는 m (mm로 저장된 STL 등은 자동 환산)."""
        vertices = np.asarray(vertices, np.float64)
        if len(faces) == 0:
            self.status.text = "메시가 비어 있습니다."
            return
        if np.ptp(vertices, axis=0).max() > 5.0:
            vertices = vertices * 0.001   # 5m보다 크면 mm 단위 파일(편집기에서 저장한 STL 등)로 봄
        self.T_align, self.floor_found = mt.floor_alignment(vertices)
        self.v = mt.transform(vertices, self.T_align)
        self.f = faces
        self.c = colors if colors is not None else np.full_like(self.v, 0.7)
        lo, hi = self.v.min(axis=0), self.v.max(axis=0)
        pad = 0.05 * (hi - lo).max()
        self.lo, self.hi = lo - pad, hi + pad
        for key, axis in (("xmin", 0), ("xmax", 0), ("ymin", 1), ("ymax", 1), ("zmin", 2), ("zmax", 2)):
            self.sliders[key].set_limits(1000 * float(self.lo[axis]), 1000 * float(self.hi[axis]))
        self.floor_warn.text = "" if self.floor_found else "바닥 평면을 못 찾아 카메라 방향으로 정렬했습니다"
        self.result = None
        self.show_result = False
        self.cb_show_result.checked = False
        self.cb_show_result.enabled = False
        self.log_label.text = ""
        self.file_label.text = name or "(이름 없음)"
        self.window.title = f"{self.title} - {name}" if name else self.title
        self._reset_box()
        self._fit_camera()
        self.status.text = f"삼각형 {len(faces):,}개. 박스를 조절한 뒤 [처리 미리보기]를 누르세요."

    def _open_dialog(self):
        if self.busy:
            return
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "스캔 열기", self.window.theme)
        dlg.add_filter(".ply .obj .stl .npz", "스캔 파일 (메시, 녹화 frames.npz)")
        dlg.add_filter(".ply .obj .stl", "메시")
        dlg.add_filter(".npz", "녹화 (정밀 재구성 다시 실행)")
        dlg.set_path(self.scans_dir)
        dlg.set_on_cancel(self.window.close_dialog)

        def ok(path):
            self.window.close_dialog()
            self.open_path(path)
        dlg.set_on_done(ok)
        self.window.show_dialog(dlg)

    def open_path(self, path):
        """메시 파일은 바로 열고, 녹화(frames.npz)는 정밀 재구성을 다시 돌린 뒤 연다.
        스캔 폴더를 주면 그 안의 scene.ply(없으면 frames.npz)."""
        path = os.path.abspath(path)
        if os.path.isdir(path):
            scene = os.path.join(path, "scene.ply")
            path = scene if os.path.exists(scene) else os.path.join(path, "frames.npz")
        if not os.path.exists(path):
            self.status.text = f"파일이 없습니다: {path}"
            return
        ext = os.path.splitext(path)[1].lower()
        folder = os.path.dirname(path)
        name = os.path.join(os.path.basename(folder), os.path.basename(path))
        app = gui.Application.instance
        self.busy = True

        def work():
            try:
                if ext in MESH_EXTS:
                    v, f, c = mt.load_mesh(path)
                elif ext == ".npz":
                    v, f, c = self._refine_recording(path, folder)
                else:
                    raise ValueError(f"지원하지 않는 파일: {ext}")
                err = None
            except Exception as e:  # noqa: BLE001 - 사용자에게 그대로 보여줌
                v = f = c = None
                err = str(e)
            app.post_to_main_thread(self.window, lambda: done(v, f, c, err))

        def done(v, f, c, err):
            self.busy = False
            if err:
                self.status.text = f"열기 실패: {err}"
                return
            self.default_dir = folder
            self.load_mesh(v, f, c, name)
            print(f"열림: {path}")

        self.status.text = "녹화로 정밀 재구성 중... (1~3분)" if ext == ".npz" else "불러오는 중..."
        threading.Thread(target=work, daemon=True).start()

    def _recorrect(self):
        """열린 스캔 폴더의 녹화로 최신 보정(마지막 연결 포함)을 다시 돌려 연다."""
        rec = os.path.join(self.default_dir, "frames.npz")
        if not os.path.exists(rec):
            self.status.text = "이 스캔 폴더에 녹화(frames.npz)가 없어 다시 보정할 수 없습니다."
            return
        self.open_path(rec)

    def _refine_recording(self, path, folder):
        """녹화로 정밀 재구성. 기존 scene.ply는 scene_prev.ply로 보관."""
        import shutil
        from scan import Recording
        from refine import Refiner, RefineConfig
        app = gui.Application.instance
        rec = Recording.load(path)

        def log(msg):
            print(msg)
            app.post_to_main_thread(self.window, lambda: setattr(self.status, "text", msg.strip()))
        cfg = RefineConfig(depth_max=2.0 if rec.subject_mode else 1.0)
        cfg.seam_ref_frames = int(self.seam_ref.int_value)
        v, f, c, _ = Refiner(rec.K, rec.scale, cfg, subject_mode=rec.subject_mode, log=log).run(
            (len(rec), rec.frame), rec.poses, rec.passes)
        scene = os.path.join(folder, "scene.ply")
        prev = os.path.join(folder, "scene_prev.ply")
        if os.path.exists(scene) and not os.path.exists(prev):
            shutil.copy2(scene, prev)
        mt.save_mesh(scene, v, f, c)
        log(f"정밀 재구성 저장: {scene}")
        return v, f, c

    # ------------------------------------------------------------ 레이아웃

    def _on_layout(self, ctx):
        r = self.window.content_rect
        pw = min(420, int(r.width * 0.35))
        self.scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    # ------------------------------------------------------------ 박스

    def _initial_box(self, found_floor):
        return mt.suggest_box(self.v, self.f, found_floor)

    def _reset_box(self):
        if self.v is None:
            return
        self.box = self._initial_box(self.floor_found)
        for k, s in self.sliders.items():
            self._sync_slider(k)
        self._invalidate_result()
        self._refresh_scene()

    def _sync_slider(self, key):
        """박스 값(m, 회전은 도) → 슬라이더(mm, 도)."""
        scale = 1.0 if key == "yaw_deg" else 1000.0
        self.sliders[key].double_value = scale * float(getattr(self.box, key))

    def _on_box(self, key, val):
        if self.v is None:
            return
        setattr(self.box, key, float(val) if key == "yaw_deg" else float(val) / 1000.0)
        # 최소값이 최대값을 넘지 않게
        for a in "xyz":
            lo, hi = a + "min", a + "max"
            if getattr(self.box, lo) > getattr(self.box, hi):
                if key == lo:
                    setattr(self.box, hi, getattr(self.box, lo))
                    self._sync_slider(hi)
                elif key == hi:
                    setattr(self.box, lo, getattr(self.box, hi))
                    self._sync_slider(lo)
        self._invalidate_result()
        self._refresh_scene()

    def _invalidate_result(self):
        if self.result is not None:
            self.result = None
            self.show_result = False
            self.cb_show_result.checked = False
            self.cb_show_result.enabled = False
            self.status.text = "박스가 바뀌었습니다. 다시 [처리 미리보기]를 누르세요."

    # ------------------------------------------------------------ 장면 그리기

    def _refresh_scene(self):
        sc = self.scene.scene
        sc.clear_geometry()
        if self.v is None:
            return
        if self.show_result and self.result is not None:
            # 결과는 mm 단위, 바닥 중심이 원점 → 표시는 m로 되돌리고 그 위치에 카메라 맞춤
            m = mt.to_o3d(self.result.vertices * 0.001, self.result.faces)
            sc.add_geometry("result", m, self.mat_result)
            bbox = m.get_axis_aligned_bounding_box()
            self.scene.setup_camera(60.0, bbox, bbox.get_center())
            return

        f_in, f_out = mt.split_by_box(self.v, self.f, self.box)
        if len(f_in):
            sc.add_geometry("inside", mt.to_o3d(self.v, f_in, self.c), self.mat_in)
        if len(f_out):
            sc.add_geometry("outside", mt.to_o3d(self.v, f_out), self.mat_out)
        sc.add_geometry("box", self.box.lineset(), self.mat_line)
        b = self.box
        self.dim_label.text = (f"박스 크기: {1000 * (b.xmax - b.xmin):.0f} × "
                               f"{1000 * (b.ymax - b.ymin):.0f} × {1000 * (b.zmax - b.zmin):.0f} mm")

    def _fit_camera(self):
        bbox = o3d.geometry.AxisAlignedBoundingBox(self.lo, self.hi)
        self.scene.setup_camera(60.0, bbox, bbox.get_center())
        c = bbox.get_center()
        ext = (self.hi - self.lo).max()
        eye = c + np.array([0.0, -1.2 * ext, 0.9 * ext])
        self.scene.look_at(c, eye, [0, 0, 1])

    # ------------------------------------------------------------ 처리 / 저장

    def _options(self):
        return mt.PrintOptions(
            largest_only=self.cb_largest.checked,
            smooth_iterations=int(self.smooth.int_value),
            target_triangles=int(self.tri.int_value),
            make_watertight=self.cb_watertight.checked,
            scale_to_mm=True)

    def _process(self, then=None):
        if self.v is None:
            self.status.text = "먼저 [파일 열기]로 스캔을 불러오세요."
            return
        self.btn_process.enabled = False
        self.status.text = "처리 중... (몇 초~수십 초)"
        box = mt.Box(**vars(self.box))
        opts = self._options()

        def work():
            try:
                res, err = mt.process_for_print(self.v, self.f, self.c, box, opts), None
            except Exception as e:  # noqa: BLE001 - 사용자에게 그대로 보여줌
                res, err = None, str(e)
            gui.Application.instance.post_to_main_thread(self.window, lambda: done(res, err))

        def done(res, err):
            self.btn_process.enabled = True
            if err:
                self.status.text = f"오류: {err}"
                return
            self.result = res
            s = res.size_mm
            vol = f", 부피 {res.volume_cm3:.1f} cm³" if res.volume_cm3 else ""
            self.status.text = f"완료: {s[0]:.1f} × {s[1]:.1f} × {s[2]:.1f} mm{vol}"
            self.log_label.text = "\n".join(res.log)
            self.cb_show_result.enabled = True
            self.cb_show_result.checked = True
            self.show_result = True
            self._refresh_scene()
            if then:
                then()

        threading.Thread(target=work, daemon=True).start()

    def _toggle_result(self, checked):
        self.show_result = checked
        self._refresh_scene()
        if not checked:
            self._fit_camera()

    def _file_dialog(self, title, filters, on_done, default_name):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, title, self.window.theme)
        for ext, desc in filters:
            dlg.add_filter(ext, desc)
        dlg.set_path(os.path.join(self.default_dir, default_name))
        dlg.set_on_cancel(self.window.close_dialog)

        def ok(path):
            self.window.close_dialog()
            on_done(path)
        dlg.set_on_done(ok)
        self.window.show_dialog(dlg)

    def _save_dialog(self):
        if self.v is None:
            self.status.text = "먼저 [파일 열기]로 스캔을 불러오세요."
            return
        if self.result is None:
            self._process(then=self._save_dialog)
            return
        self._file_dialog("저장", [(".stl", "STL (3D 프린팅)"), (".obj", "OBJ"), (".ply", "PLY")],
                          self._save_result, "model.stl")

    def _save_result(self, path):
        if not os.path.splitext(path)[1]:
            path += ".stl"
        r = self.result
        try:
            mt.save_mesh(path, r.vertices, r.faces, r.colors)
            self.status.text = f"저장됨: {path}"
            print(f"저장됨: {path}")
        except Exception as e:  # noqa: BLE001
            self.status.text = f"저장 실패: {e}"

    def _open_repair(self):
        if self.v is None:
            self.status.text = "먼저 [파일 열기]로 스캔을 불러오세요."
            return
        from repair import RepairWindow
        base_name = self.file_label.text

        def done(v, f, c):
            self.load_mesh(v, f, c, f"{base_name} (다듬음)")
            self.status.text = "다듬은 결과를 불러왔습니다. 필요하면 [원본 장면 저장]으로 보관하세요."

        # 버튼 콜백 안에서 새 창을 바로 만들면 Open3D가 통째로 꺼진다(0xc000041d, 재현 확인).
        # 콜백이 끝난 뒤 메인 루프에서 만들도록 예약만 한다.
        v, f, c, d = self.v, self.f, self.c, self.default_dir
        self.status.text = "다듬기 도구 여는 중..."
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: RepairWindow(v, f, c, on_done=done, default_dir=d))

    def _save_scene_dialog(self):
        if self.v is None:
            return
        self._file_dialog("원본 장면 저장", [(".ply", "PLY")], self._save_scene, "scene.ply")

    def _save_scene(self, path):
        if not os.path.splitext(path)[1]:
            path += ".ply"
        try:
            mt.save_mesh(path, self.v, self.f, self.c)
            self.status.text = f"저장됨: {path}"
        except Exception as e:  # noqa: BLE001
            self.status.text = f"저장 실패: {e}"


def _ascii_resource_path():
    """Open3D GUI는 리소스 경로에 한글이 있으면 초기화에 실패한다.
    (venv가 '바탕 화면' 아래에 있음) → 영문 경로로 복사해서 사용."""
    import shutil
    src = os.path.join(os.path.dirname(o3d.__file__), "resources")
    if src.isascii():
        return src
    dst = os.path.join(os.environ.get("LOCALAPPDATA", r"C:\ProgramData"),
                       "rs_scanner", f"o3d_resources_{o3d.__version__}")
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)
    return dst


def init_app():
    import gui_safety
    gui_safety.install()   # 화면 콜백 오류로 프로그램이 통째로 꺼지지 않게
    app = gui.Application.instance
    app.initialize(_ascii_resource_path())
    if os.path.exists(KOREAN_FONT):
        font = gui.FontDescription()
        font.add_typeface_for_language(KOREAN_FONT, "ko")
        app.set_font(app.DEFAULT_FONT_ID, font)
    return app


def run_editor(vertices=None, faces=None, colors=None, default_dir=".", name="", scans_dir=None,
               open_path=None):
    """vertices가 없으면 빈 편집기를 띄운다. open_path가 있으면 그 파일(메시/녹화/스캔 폴더)을 열고,
    없으면 파일 선택 창을 연다."""
    app = init_app()
    ed = Editor(vertices, faces, colors, default_dir=default_dir, name=name, scans_dir=scans_dir,
                open_on_start=open_path is None)
    if open_path:
        ed.open_path(open_path)
    app.run()
