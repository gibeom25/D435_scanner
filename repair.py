"""다듬기 도구: 따로 떨어진 조각을 이어 붙이고, 안 찍힌 구멍을 예측해서 메운다.

1. 조각 잇기
   - 메시를 연결된 덩어리(조각)로 나눈다. 가장 큰 조각이 기준.
   - 조각을 슬라이더로 원하는 위치·방향으로 옮긴 뒤 [자동 맞추기](ICP)로 기준에 딱 붙인다.
     위치를 전혀 모르면 [자동으로 위치 찾기](3D 형태 기술자 FPFH + RANSAC).
2. 구멍 메우기
   - 남은 조각 전체로 Poisson 표면 재구성 → 주변 곡면을 따라 빈 곳까지 이어진 표면을 예측.
   - 실제 데이터에서 먼 부분 = 예측한 부분 (주황색으로 표시).
   - 메울 최대 거리, 볼록함(오목/평평/볼록), 매끄럽게를 슬라이더로 바로 조절.
"""
import os
import threading
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy import sparse
from scipy.spatial import cKDTree

import mesh_tools as mt
from sculpt import Sculpt

reg = o3d.pipelines.registration


# ====================================================================== 계산

@dataclass
class Piece:
    v: np.ndarray          # 원래 좌표 (m)
    f: np.ndarray
    c: np.ndarray
    pose: np.ndarray       # 확정된 변환 (4x4)
    keep: bool = True

    @property
    def tris(self):
        return len(self.f)

    def world(self, delta=None):
        T = self.pose if delta is None else delta @ self.pose
        return self.v @ T[:3, :3].T + T[:3, 3]


def split_pieces(v, f, c, min_ratio=0.01, min_tris=300):
    """연결된 덩어리로 분리. 큰 순서로 정렬. 아주 작은 부스러기는 keep=False."""
    m = mt.to_o3d(v, f)
    labels, counts, _ = m.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    order = np.argsort(-counts)
    pieces = []
    for lab in order:
        tri = f[labels == lab]
        used, inv = np.unique(tri, return_inverse=True)
        p = Piece(v[used], inv.reshape(-1, 3), c[used], np.eye(4))
        p.keep = counts[lab] >= max(min_tris, min_ratio * counts[order[0]])
        pieces.append(p)
    return pieces


def _pcd(v, f, voxel):
    m = mt.to_o3d(v, f)
    pcd = o3d.geometry.PointCloud(m.vertices)
    pcd.normals = m.vertex_normals
    return pcd.voxel_down_sample(voxel)


def snap_icp(src_v, src_f, dst_v, dst_f, init=np.eye(4)):
    """src를 dst에 맞추는 변환 (거친 → 세밀 3단계 ICP). 반환: (T, 겹침 비율, 오차 mm)."""
    src, dst = _pcd(src_v, src_f, 0.003), _pcd(dst_v, dst_f, 0.003)
    T = init
    res = None
    for dist in (0.03, 0.015, 0.007):
        res = reg.registration_icp(src, dst, dist, T,
                                   reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=dist)),
                                   reg.ICPConvergenceCriteria(max_iteration=40))
        T = res.transformation
    return T, res.fitness, res.inlier_rmse * 1000


def find_global(src_v, src_f, dst_v, dst_f):
    """위치를 모를 때: FPFH + RANSAC으로 대략 찾고 ICP로 다듬는다."""
    voxel = 0.006
    src, dst = _pcd(src_v, src_f, voxel), _pcd(dst_v, dst_f, voxel)
    param = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
    fs, fd = reg.compute_fpfh_feature(src, param), reg.compute_fpfh_feature(dst, param)
    r = reg.registration_ransac_based_on_feature_matching(
        src, dst, fs, fd, True, voxel * 1.5, reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        reg.RANSACConvergenceCriteria(200000, 0.999))
    return snap_icp(src_v, src_f, dst_v, dst_f, r.transformation)


@dataclass
class FillBase:
    """Poisson 결과(한 번 계산) + 각 꼭짓점이 실제 데이터에서 떨어진 거리. 슬라이더 조절은 이걸로 즉시."""
    v: np.ndarray
    f: np.ndarray
    n: np.ndarray
    c: np.ndarray
    dist: np.ndarray       # 실제 데이터까지 거리 (m)
    adj: sparse.csr_matrix  # 이웃 평균용 행렬


def poisson_base(points, normals, colors, depth=9):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=1.1)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles).astype(np.int64)
    tree = cKDTree(points)
    dist, idx = tree.query(v)
    n = len(v)
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    adj = sparse.diags(1.0 / np.maximum(deg, 1)) @ A
    return FillBase(v, f, np.asarray(mesh.vertex_normals), colors[idx], dist, adj.tocsr())


def shape_fill(base: FillBase, max_fill_m, bulge, smooth_iters, data_tol_m=0.003):
    """예측 영역을 조절해 최종 메시를 만든다.
    max_fill_m: 실제 데이터에서 이보다 먼 부분은 버림 (0이면 거의 안 메움)
    bulge: -1(오목) ~ 0(Poisson 그대로) ~ +1(볼록). 데이터에서 멀수록 법선 방향으로 더 밀어냄
    반환: (v, f, c, predicted_mask)"""
    v = base.v.copy()
    pred = base.dist > data_tol_m
    if bulge:
        off = bulge * 0.5 * np.clip(base.dist - data_tol_m, 0, None)
        v += base.n * off[:, None]
    for _ in range(int(smooth_iters)):
        v[pred] = (base.adj @ v)[pred]
    keep_v = base.dist <= max(max_fill_m, data_tol_m)
    keep_f = keep_v[base.f].all(axis=1)
    f = base.f[keep_f]
    used, inv = np.unique(f, return_inverse=True)
    return v[used], inv.reshape(-1, 3), base.c[used], pred[used]


# ====================================================================== 화면

class RepairWindow:
    """조각 잇기 + 구멍 메우기. on_done(v, f, c)로 결과를 넘긴다(없으면 저장만)."""

    def __init__(self, vertices, faces, colors, on_done=None, title="다듬기 도구", default_dir="."):
        self.on_done = on_done
        self.default_dir = default_dir
        colors = colors if colors is not None else np.full_like(vertices, 0.7)
        self.pieces = split_pieces(np.asarray(vertices, float), np.asarray(faces), colors)
        self.sel = 0
        self.delta_vals = dict(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
        self.fill_base = None
        self.fill_result = None
        self.mode = "pieces"     # "pieces" | "fill"
        self.busy = False
        self.history = []        # 되돌리기: (조각 번호, 이전 pose, 이전 keep)
        self.sculpt = None       # 끌어서 모양 다듬기 상태
        self.drag = None         # 끄는 중: dict(anchor, plane_n, static_added)
        self.push_base = 0.0

        app = gui.Application.instance
        self.window = app.create_window(title, 1500, 950)
        w = self.window
        em = w.theme.font_size
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.12, 0.12, 0.14, 1.0])
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.mat_off = rendering.MaterialRecord()
        self.mat_off.shader = "defaultLitTransparency"
        self.mat_off.base_color = [0.5, 0.5, 0.55, 0.2]
        self.mat_line = rendering.MaterialRecord()
        self.mat_line.shader = "unlitLine"
        self.mat_line.line_width = 2.5

        panel = gui.Vert(0.35 * em, gui.Margins(em, em, em, em))

        # ---------- 1. 조각 잇기
        panel.add_child(gui.Label("1. 조각 잇기"))
        self.cb_small = gui.Checkbox("작은 부스러기도 보기")
        self.cb_small.set_on_checked(lambda _: self._refresh_list())
        panel.add_child(self.cb_small)
        self.list = gui.ListView()
        self.list.set_max_visible_items(6)
        self.list.set_on_selection_changed(self._on_select)
        panel.add_child(self.list)

        grid = gui.VGrid(2, 0.25 * em)
        self.sl = {}
        for key, label, lo, hi in (("tx", "이동 X (mm)", -300, 300), ("ty", "이동 Y (mm)", -300, 300),
                                   ("tz", "이동 Z (mm)", -300, 300), ("rx", "회전 X (도)", -180, 180),
                                   ("ry", "회전 Y (도)", -180, 180), ("rz", "회전 Z (도)", -180, 180)):
            s = gui.Slider(gui.Slider.DOUBLE)
            s.set_limits(lo, hi)
            s.set_on_value_changed(lambda val, k=key: self._on_move(k, val))
            grid.add_child(gui.Label(label))
            grid.add_child(s)
            self.sl[key] = s
        panel.add_child(grid)

        row = gui.Horiz(0.25 * em)
        for text, cb in (("자동 맞추기", self._snap), ("자동으로 위치 찾기", self._find)):
            b = gui.Button(text)
            b.set_on_clicked(cb)
            row.add_child(b)
        panel.add_child(row)
        row = gui.Horiz(0.25 * em)
        self.btn_keep = gui.Button("이 조각 버리기")
        self.btn_keep.set_on_clicked(self._toggle_keep)
        row.add_child(self.btn_keep)
        b = gui.Button("처음 위치로")
        b.set_on_clicked(self._reset_piece)
        row.add_child(b)
        b = gui.Button("되돌리기")
        b.set_on_clicked(self._undo)
        row.add_child(b)
        panel.add_child(row)
        hint = gui.Label("팁: 슬라이더로 대충 가까이 놓고 [자동 맞추기]가 가장 확실합니다.")
        hint.text_color = gui.Color(0.6, 0.6, 0.6)
        panel.add_child(hint)

        # ---------- 2. 구멍 메우기
        panel.add_fixed(em)
        panel.add_child(gui.Label("2. 구멍 메우기 (예측 부분은 주황색)"))
        grid2 = gui.VGrid(2, 0.25 * em)
        self.depth = gui.Slider(gui.Slider.INT)
        self.depth.set_limits(7, 10)
        self.depth.int_value = 9
        grid2.add_child(gui.Label("세밀도"))
        grid2.add_child(self.depth)
        self.max_fill = gui.Slider(gui.Slider.DOUBLE)
        self.max_fill.set_limits(0, 150)
        self.max_fill.double_value = 40
        self.max_fill.set_on_value_changed(lambda _: self._update_fill())
        grid2.add_child(gui.Label("메울 최대 거리 (mm)"))
        grid2.add_child(self.max_fill)
        self.bulge = gui.Slider(gui.Slider.DOUBLE)
        self.bulge.set_limits(-1, 1)
        self.bulge.double_value = 0
        self.bulge.set_on_value_changed(lambda _: self._update_fill())
        grid2.add_child(gui.Label("볼록함 (-오목 / +볼록)"))
        grid2.add_child(self.bulge)
        self.smooth = gui.Slider(gui.Slider.INT)
        self.smooth.set_limits(0, 30)
        self.smooth.int_value = 3
        self.smooth.set_on_value_changed(lambda _: self._update_fill())
        grid2.add_child(gui.Label("매끄럽게"))
        grid2.add_child(self.smooth)
        panel.add_child(grid2)
        row = gui.Horiz(0.25 * em)
        b = gui.Button("구멍 메우기 미리보기")
        b.set_on_clicked(self._make_fill)
        row.add_child(b)
        b = gui.Button("조각 보기로")
        b.set_on_clicked(self._back_to_pieces)
        row.add_child(b)
        panel.add_child(row)

        # ---------- 3. 끌어서 모양 다듬기
        panel.add_fixed(em)
        panel.add_child(gui.Label("3. 끌어서 모양 다듬기"))
        row = gui.Horiz(0.25 * em)
        self.btn_sculpt = gui.Button("모양 다듬기 시작")
        self.btn_sculpt.set_on_clicked(self._toggle_sculpt)
        row.add_child(self.btn_sculpt)
        b = gui.Button("선택 해제")
        b.set_on_clicked(self._sculpt_clear)
        row.add_child(b)
        b = gui.Button("모양 되돌리기")
        b.set_on_clicked(self._sculpt_undo)
        row.add_child(b)
        panel.add_child(row)
        grid3 = gui.VGrid(2, 0.25 * em)
        self.brush = gui.Slider(gui.Slider.DOUBLE)
        self.brush.set_limits(2, 60)
        self.brush.double_value = 10
        grid3.add_child(gui.Label("붓 크기 (mm)"))
        grid3.add_child(self.brush)
        self.radius = gui.Slider(gui.Slider.DOUBLE)
        self.radius.set_limits(0, 150)
        self.radius.double_value = 40
        self.radius.set_on_value_changed(self._on_radius)
        grid3.add_child(gui.Label("영향 반경 (mm)"))
        grid3.add_child(self.radius)
        self.push = gui.Slider(gui.Slider.DOUBLE)
        self.push.set_limits(-30, 30)
        self.push.double_value = 0
        self.push.set_on_value_changed(self._on_push)
        grid3.add_child(gui.Label("표면 방향 밀기/당기기 (mm)"))
        grid3.add_child(self.push)
        panel.add_child(grid3)
        # 가우시안 스무딩 (표면을 따라 퍼짐)
        grid4 = gui.VGrid(2, 0.25 * em)
        self.gsigma = gui.Slider(gui.Slider.DOUBLE)
        self.gsigma.set_limits(0.5, 10)
        self.gsigma.double_value = 2.0
        grid4.add_child(gui.Label("가우시안 세기 (mm)"))
        grid4.add_child(self.gsigma)
        panel.add_child(grid4)
        self.cb_keepvol = gui.Checkbox("부피 유지 (쪼그라들지 않게)")
        self.cb_keepvol.checked = True
        panel.add_child(self.cb_keepvol)
        row = gui.Horiz(0.25 * em)
        b = gui.Button("선택 부분 스무딩")
        b.set_on_clicked(lambda: self._gauss(True))
        row.add_child(b)
        b = gui.Button("전체 스무딩")
        b.set_on_clicked(lambda: self._gauss(False))
        row.add_child(b)
        panel.add_child(row)
        hint = gui.Label("Shift+클릭: 선택 추가  |  Alt+클릭: 선택 빼기\n"
                         "Ctrl+드래그: 선택 부분 끌기 (주변은 표면 따라 부드럽게 따라옴)\n"
                         "그냥 드래그: 화면 회전")
        hint.text_color = gui.Color(0.6, 0.6, 0.6)
        panel.add_child(hint)

        # ---------- 4. 끝내기
        panel.add_fixed(em)
        row = gui.Horiz(0.25 * em)
        b = gui.Button("결과 사용" if on_done else "결과 저장")
        b.set_on_clicked(self._finish)
        row.add_child(b)
        b = gui.Button("PLY로 저장")
        b.set_on_clicked(self._save_dialog)
        row.add_child(b)
        panel.add_child(row)
        self.status = gui.Label("")
        panel.add_child(self.status)

        self.panel = panel
        w.add_child(self.scene)
        w.add_child(panel)
        w.set_on_layout(self._on_layout)
        self.scene.set_on_mouse(self._on_mouse)
        import gui_safety
        gui_safety.on_error = lambda msg: setattr(self.status, "text", msg + " (프로그램은 계속 동작)")
        self._refresh_list()
        self._refresh()
        self._fit()

    # ------------------------------------------------------------ 공통

    def _on_layout(self, ctx):
        r = self.window.content_rect
        pw = min(440, int(r.width * 0.35))
        self.scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def _fit(self):
        pts = np.concatenate([p.world() for p in self.pieces if p.keep] or [self.pieces[0].world()])
        bbox = o3d.geometry.AxisAlignedBoundingBox(pts.min(0), pts.max(0))
        self.scene.setup_camera(60.0, bbox, bbox.get_center())

    def _run(self, msg, work, done):
        """무거운 계산은 별도 스레드에서, 결과 반영은 화면 스레드에서."""
        if self.busy:
            return
        self.busy = True
        self.status.text = msg
        app = gui.Application.instance

        def wrap():
            try:
                out, err = work(), None
            except Exception as e:  # noqa: BLE001
                out, err = None, str(e)

            def fin():
                self.busy = False
                if err:
                    self.status.text = f"오류: {err}"
                else:
                    done(out)
            app.post_to_main_thread(self.window, fin)
        threading.Thread(target=wrap, daemon=True).start()

    # ------------------------------------------------------------ 조각

    def _push(self, i=None):
        i = self.sel if i is None else i
        self.history.append((i, self.pieces[i].pose.copy(), self.pieces[i].keep))
        del self.history[:-50]

    def _undo(self):
        self._bake()
        if not self.history:
            self.status.text = "되돌릴 작업이 없습니다."
            return
        i, pose, keep = self.history.pop()
        self.pieces[i].pose, self.pieces[i].keep = pose, keep
        self.sel = i
        self.fill_base = None
        self.mode = "pieces"
        self._refresh_list()
        self._refresh()
        self.status.text = "되돌렸습니다."

    def _refresh_list(self):
        self._bake()
        self.vis_ids = [i for i, p in enumerate(self.pieces)
                        if self.cb_small.checked or p.keep or p.tris >= 300]
        items = []
        for i in self.vis_ids:
            p = self.pieces[i]
            tag = "기준" if i == 0 else f"조각 {i}"
            items.append(f"{tag}: 삼각형 {p.tris:,}{'' if p.keep else '  (버림)'}")
        self.list.set_items(items)
        if self.sel in self.vis_ids:
            self.list.selected_index = self.vis_ids.index(self.sel)

    def _on_select(self, value, is_dbl):
        idx = self.list.selected_index
        if 0 <= idx < len(self.vis_ids):
            self._bake()
            self.sel = self.vis_ids[idx]
            self.btn_keep.text = "이 조각 버리기" if self.pieces[self.sel].keep else "이 조각 다시 쓰기"
            self._refresh()

    def _delta(self):
        """슬라이더 값 → 선택 조각 중심 기준 변환."""
        d = self.delta_vals
        p = self.pieces[self.sel]
        center = p.world().mean(axis=0)
        R = o3d.geometry.get_rotation_matrix_from_xyz(np.deg2rad([d["rx"], d["ry"], d["rz"]]))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center - R @ center + np.array([d["tx"], d["ty"], d["tz"]]) / 1000.0
        return T

    def _bake(self):
        """슬라이더로 옮긴 만큼을 조각 위치에 확정하고 슬라이더를 0으로."""
        if any(self.delta_vals.values()):
            p = self.pieces[self.sel]
            self._push()
            p.pose = self._delta() @ p.pose
            for k in self.delta_vals:
                self.delta_vals[k] = 0.0
        for s in self.sl.values():
            s.double_value = 0.0

    def _on_move(self, key, val):
        self.delta_vals[key] = float(val)
        self.fill_base = None
        self._refresh()

    def _others(self):
        vs, fs, off = [], [], 0
        for i, p in enumerate(self.pieces):
            if i != self.sel and p.keep:
                vs.append(p.world())
                fs.append(p.f + off)
                off += len(p.v)
        if not vs:
            raise ValueError("맞출 기준이 없습니다 (다른 조각이 모두 버려짐).")
        return np.concatenate(vs), np.concatenate(fs)

    def _snap(self):
        if self.sel == 0 and len(self.pieces) > 1:
            pass  # 기준 조각도 다른 조각들에 맞출 수 있음
        self._bake()
        p = self.pieces[self.sel]

        def work():
            dv, df = self._others()
            return snap_icp(p.world(), p.f, dv, df)

        def done(out):
            T, fit, rmse = out
            if fit < 0.05:
                self.status.text = f"겹치는 부분이 거의 없습니다 (겹침 {fit:.0%}). 더 가깝게 옮긴 뒤 다시 시도하세요."
                return
            self._push()
            p.pose = T @ p.pose
            self.fill_base = None
            self.status.text = f"맞춤 완료: 겹침 {fit:.0%}, 오차 {rmse:.1f}mm"
            self._refresh()
        self._run("자동 맞추기 중...", work, done)

    def _find(self):
        self._bake()
        p = self.pieces[self.sel]

        def work():
            dv, df = self._others()
            return find_global(p.world(), p.f, dv, df)

        def done(out):
            T, fit, rmse = out
            if fit < 0.1:
                self.status.text = f"확실한 위치를 못 찾았습니다 (겹침 {fit:.0%}). 손으로 가까이 옮긴 뒤 [자동 맞추기]를 쓰세요."
                return
            self._push()
            p.pose = T @ p.pose
            self.fill_base = None
            self.status.text = (f"위치 찾음: 겹침 {fit:.0%}, 오차 {rmse:.1f}mm. "
                                "엉뚱하면 [되돌리기] 후 손으로 가까이 놓고 [자동 맞추기]")
            self._refresh()
        self._run("위치 찾는 중... (수십 초)", work, done)

    def _toggle_keep(self):
        self._bake()
        p = self.pieces[self.sel]
        self._push()
        p.keep = not p.keep
        self.fill_base = None
        self._refresh_list()
        self.btn_keep.text = "이 조각 버리기" if p.keep else "이 조각 다시 쓰기"
        self._refresh()

    def _reset_piece(self):
        for k in self.delta_vals:
            self.delta_vals[k] = 0.0
        for s in self.sl.values():
            s.double_value = 0.0
        self._push()
        self.pieces[self.sel].pose = np.eye(4)
        self.fill_base = None
        self._refresh()

    # ------------------------------------------------------------ 구멍 메우기

    def _merged_points(self):
        self._bake()
        pts, nrm, col = [], [], []
        for p in self.pieces:
            if not p.keep:
                continue
            m = mt.to_o3d(p.world(), p.f, p.c)
            pts.append(np.asarray(m.vertices))
            nrm.append(np.asarray(m.vertex_normals))
            col.append(np.asarray(m.vertex_colors))
        if not pts:
            raise ValueError("남은 조각이 없습니다.")
        return np.concatenate(pts), np.concatenate(nrm), np.concatenate(col)

    def _make_fill(self):
        if self.mode == "sculpt":
            self.status.text = "모양 다듬기 중입니다. 먼저 [모양 다듬기 끝]을 누르세요."
            return
        depth = int(self.depth.int_value)

        def work():
            P, N, C = self._merged_points()
            return poisson_base(P, N, C, depth)

        def done(base):
            self.fill_base = base
            self.mode = "fill"
            self._update_fill()
        self._run("표면 예측 중... (수 초~수십 초)", work, done)

    def _update_fill(self):
        if self.mode != "fill" or self.fill_base is None:
            return
        v, f, c, pred = shape_fill(self.fill_base, self.max_fill.double_value / 1000.0,
                                   self.bulge.double_value, self.smooth.int_value)
        self.fill_result = (v, f, c)
        show = c.copy()
        show[pred] = 0.5 * show[pred] + 0.5 * np.array([1.0, 0.55, 0.1])
        sc = self.scene.scene
        sc.clear_geometry()
        sc.add_geometry("fill", mt.to_o3d(v, f, show), self.mat)
        self.status.text = f"삼각형 {len(f):,}개, 예측한 부분 {pred.mean():.0%}"

    def _back_to_pieces(self):
        if self.mode == "sculpt":
            self._toggle_sculpt()
        self.mode = "pieces"
        self._refresh()

    # ------------------------------------------------------------ 그리기

    def _refresh(self):
        if self.mode == "sculpt" and self.sculpt is not None:
            self._draw_sculpt()
            return
        if self.mode == "fill" and self.fill_base is not None:
            self._update_fill()
            return
        self.mode = "pieces"
        sc = self.scene.scene
        sc.clear_geometry()
        delta = self._delta()
        for i, p in enumerate(self.pieces):
            if not (p.keep or i == self.sel or self.cb_small.checked):
                continue
            v = p.world(delta if i == self.sel else None)
            sc.add_geometry(f"p{i}", mt.to_o3d(v, p.f, p.c), self.mat if p.keep else self.mat_off)
            if i == self.sel:
                box = o3d.geometry.AxisAlignedBoundingBox(v.min(0), v.max(0))
                ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
                ls.paint_uniform_color([1.0, 0.55, 0.0])
                sc.add_geometry("sel", ls, self.mat_line)
        kept = sum(p.keep for p in self.pieces)
        if not self.busy:
            self.status.text = f"조각 {len(self.pieces)}개 중 {kept}개 사용 중. 주황 상자 = 선택한 조각"


    # ------------------------------------------------------------ 끌어서 모양 다듬기

    def _toggle_sculpt(self):
        if self.mode == "sculpt":
            # 끝: 다듬은 모양을 하나의 조각으로 만들어 다음 작업(구멍 메우기 등)에 이어 쓴다
            self._commit_push()
            sc = self.sculpt
            p = Piece(sc.base.copy(), sc.f.copy(), sc.c.copy(), np.eye(4))
            self.pieces = [p]
            self.sel = 0
            self.history = []
            self.sculpt = None
            self.fill_base = None
            self.mode = "pieces"
            self.btn_sculpt.text = "모양 다듬기 시작"
            self._refresh_list()
            self._refresh()
            self.status.text = "다듬은 모양을 확정했습니다."
            return
        v, f, c = self._current_result()
        self.status.text = "모양 다듬기 준비 중..."

        def work():
            return Sculpt(v, f, c)

        def done(sc):
            self.sculpt = sc
            sc.set_radius(self.radius.double_value / 1000.0)
            self.mode = "sculpt"
            self.btn_sculpt.text = "모양 다듬기 끝"
            self.push.double_value = 0
            self.push_base = 0.0
            self._rebuild_pick()
            self._draw_sculpt()
            self.status.text = "Shift+클릭으로 끌어올 부분을 선택하세요."
        self._run("모양 다듬기 준비 중...", work, done)

    def _rebuild_pick(self):
        sc = self.sculpt
        self.pick = o3d.t.geometry.RaycastingScene()
        self.pick.add_triangles(o3d.core.Tensor(sc.base.astype(np.float32)),
                                o3d.core.Tensor(sc.f.astype(np.uint32)))

    def _draw_sculpt(self, positions=None):
        sc = self.sculpt
        v = sc.base if positions is None else positions
        scene = self.scene.scene
        scene.clear_geometry()
        scene.add_geometry("sculpt", mt.to_o3d(v, sc.f, sc.display_colors()), self.mat)

    def _ray(self, x, y):
        """화면 좌표(위젯 기준) → 월드 광선 (원점, 방향)."""
        fr = self.scene.frame
        cam = self.scene.scene.camera
        p0 = np.asarray(cam.unproject(x, y, 0.0, fr.width, fr.height))
        p1 = np.asarray(cam.unproject(x, y, 0.5, fr.width, fr.height))
        d = p1 - p0
        return p0, d / (np.linalg.norm(d) + 1e-12)

    def pick_point(self, x, y):
        o, d = self._ray(x, y)
        rays = o3d.core.Tensor(np.concatenate([o, d])[None].astype(np.float32))
        t = self.pick.cast_rays(rays)["t_hit"].numpy()[0]
        return None if not np.isfinite(t) else o + d * t

    def _cam_forward(self):
        V = np.asarray(self.scene.scene.camera.get_view_matrix())
        f = -V[2, :3]
        return f / (np.linalg.norm(f) + 1e-12)

    def _on_mouse(self, e):
        if self.mode != "sculpt" or self.sculpt is None or self.busy:
            return gui.Widget.EventCallbackResult.IGNORED
        x, y = e.x - self.scene.frame.x, e.y - self.scene.frame.y
        shift = e.is_modifier_down(gui.KeyModifier.SHIFT)
        ctrl = e.is_modifier_down(gui.KeyModifier.CTRL)
        alt = e.is_modifier_down(gui.KeyModifier.ALT)
        T = gui.MouseEvent.Type
        if e.type == T.BUTTON_DOWN and (shift or alt):
            self.select_at(x, y, add=not alt)
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.type == T.BUTTON_DOWN and ctrl:
            self.drag_start(x, y)
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.type == T.DRAG and self.drag is not None:
            self.drag_move(x, y)
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.type == T.BUTTON_UP and self.drag is not None:
            self.drag_end(x, y)
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

    def select_at(self, x, y, add=True):
        self._commit_push()
        p = self.pick_point(x, y)
        if p is None:
            self.status.text = "메시 위를 클릭하세요."
            return
        n = self.sculpt.select(p, self.brush.double_value / 1000.0, add)
        self._draw_sculpt()
        k = int(self.sculpt.sel.sum())
        self.status.text = f"선택 {k:,}개 꼭짓점 ({'추가' if add else '빼기'} {n:,}). Ctrl+드래그로 끌어오세요."

    def drag_start(self, x, y):
        sc = self.sculpt
        self._commit_push()
        if not sc.sel.any():
            self.status.text = "먼저 Shift+클릭으로 끌어올 부분을 선택하세요."
            return
        anchor = sc.base[sc.sel].mean(axis=0)
        n = self._cam_forward()
        # 끄는 동안은 영향받는 부분만 다시 그린다 (나머지는 고정 메시로 한 번만)
        aff = sc.affected()
        tri_dyn = aff[sc.f].any(axis=1)
        scene = self.scene.scene
        scene.clear_geometry()
        col = sc.display_colors()
        stat = sc.f[~tri_dyn]
        if len(stat):
            scene.add_geometry("static", mt.to_o3d(sc.base, stat, col), self.mat)
        self.drag = dict(anchor=anchor, n=n, start=self._plane_hit(x, y, anchor, n),
                         dyn=sc.f[tri_dyn], col=col, move=np.zeros(3))

    def _plane_hit(self, x, y, anchor, n):
        o, d = self._ray(x, y)
        den = d @ n
        if abs(den) < 1e-9:
            return anchor
        return o + d * ((anchor - o) @ n / den)

    def drag_move(self, x, y):
        dr = self.drag
        if dr is None or dr["start"] is None:
            return
        move = self._plane_hit(x, y, dr["anchor"], dr["n"]) - dr["start"]
        dr["move"] = move
        pos = self.sculpt.preview(move)
        scene = self.scene.scene
        if scene.has_geometry("dyn"):
            scene.remove_geometry("dyn")
        scene.add_geometry("dyn", mt.to_o3d(pos, dr["dyn"], dr["col"]), self.mat)
        self.status.text = f"끌기: {np.linalg.norm(move) * 1000:.1f} mm"

    def drag_end(self, x, y):
        dr, self.drag = self.drag, None
        if dr is None:
            return
        self.sculpt.commit(dr["move"])
        self._rebuild_pick()
        self._draw_sculpt()
        self.status.text = f"{np.linalg.norm(dr['move']) * 1000:.1f} mm 끌었습니다. [모양 되돌리기]로 취소."

    def _on_radius(self, val):
        if self.sculpt is None:
            return
        self._commit_push()
        self.sculpt.set_radius(val / 1000.0)
        self._draw_sculpt()

    def _on_push(self, val):
        """표면 방향으로 밀기/당기기: 슬라이더를 움직이는 동안 미리보기, 다른 조작 때 확정."""
        sc = self.sculpt
        if sc is None or not sc.sel.any():
            return
        move = sc.selection_normal() * (val - self.push_base) / 1000.0
        self._draw_sculpt(sc.preview(move))

    def _commit_push(self):
        sc = self.sculpt
        if sc is None:
            return
        val = self.push.double_value
        if val != self.push_base and sc.sel.any():
            sc.commit(sc.selection_normal() * (val - self.push_base) / 1000.0)
            self._rebuild_pick()
        self.push_base = val

    def _gauss(self, selection_only):
        """가우시안 스무딩. 모양 다듬기 중일 때 그 메시에 적용 ([모양 되돌리기]로 취소)."""
        sc = self.sculpt
        if self.mode != "sculpt" or sc is None:
            self.status.text = "먼저 [모양 다듬기 시작]을 누르세요 (스무딩은 그 메시에 적용됩니다)."
            return
        self._commit_push()
        if selection_only and not sc.sel.any():
            self.status.text = "선택한 부분이 없습니다. Shift+클릭으로 선택하거나 [전체 스무딩]을 쓰세요."
            return
        sigma = self.gsigma.double_value / 1000.0
        keep = self.cb_keepvol.checked

        def work():
            return sc.gaussian_smooth(sigma, selection_only=selection_only, keep_volume=keep)

        def done(moved):
            self._rebuild_pick()
            self._draw_sculpt()
            where = "선택 부분" if selection_only else "전체"
            self.status.text = (f"{where} 가우시안 스무딩 세기 {sigma * 1000:.1f}mm (반복 {sc.last_iters}회, "
                                f"꼭짓점 {moved:,}개). [모양 되돌리기]로 취소")
        self._run("스무딩 중...", work, done)

    def _sculpt_clear(self):
        if self.sculpt is None:
            return
        self._commit_push()
        self.sculpt.clear()
        self._draw_sculpt()

    def _sculpt_undo(self):
        if self.sculpt is None:
            return
        self._commit_push()
        if self.sculpt.undo():
            self._rebuild_pick()
            self._draw_sculpt()
            self.status.text = "모양을 되돌렸습니다."
        else:
            self.status.text = "되돌릴 모양 변경이 없습니다."

    # ------------------------------------------------------------ 결과

    def _current_result(self):
        if self.mode == "sculpt" and self.sculpt is not None:
            self._commit_push()
            return self.sculpt.base.copy(), self.sculpt.f.copy(), self.sculpt.c.copy()
        if self.mode == "fill" and self.fill_result is not None:
            return self.fill_result
        self._bake()
        vs, fs, cs, off = [], [], [], 0
        for p in self.pieces:
            if p.keep:
                vs.append(p.world())
                fs.append(p.f + off)
                cs.append(p.c)
                off += len(p.v)
        return np.concatenate(vs), np.concatenate(fs), np.concatenate(cs)

    def _finish(self):
        v, f, c = self._current_result()
        if self.on_done:
            # 콜백 안에서 창을 닫거나 다른 창을 바꾸지 않고, 콜백이 끝난 뒤 처리 (창 생성과 같은 이유)
            cb = self.on_done

            def go():
                cb(v, f, c)
                self.window.close()
            gui.Application.instance.post_to_main_thread(self.window, go)
        else:
            self._save_dialog()

    def _save_dialog(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "저장", self.window.theme)
        dlg.add_filter(".ply", "PLY (색 포함)")
        dlg.set_path(os.path.join(self.default_dir, "scene_repaired.ply"))
        dlg.set_on_cancel(self.window.close_dialog)

        def ok(path):
            self.window.close_dialog()
            if not path.lower().endswith(".ply"):
                path += ".ply"
            v, f, c = self._current_result()
            mt.save_mesh(path, v, f, c)
            self.status.text = f"저장됨: {path}"
            print(f"저장됨: {path}")
        dlg.set_on_done(ok)
        self.window.show_dialog(dlg)


def run_repair(path, default_dir="."):
    """단독 실행: 파일을 열어 다듬기 창을 띄운다 (결과는 PLY로 저장)."""
    from editor import init_app
    app = init_app()
    v, f, c = mt.load_mesh(path)
    if np.ptp(v, axis=0).max() > 5.0:
        v = v * 0.001
    RepairWindow(v, f, c, title=f"다듬기 도구 - {os.path.basename(path)}", default_dir=default_dir)
    app.run()
