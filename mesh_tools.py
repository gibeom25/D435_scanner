"""메시 후처리: 바닥 정렬, 박스 자르기, 3D 프린팅용 정리, 저장.

파일 입출력은 전부 trimesh(파이썬 open)로 한다. Open3D/OpenCV의 파일 함수는
Windows에서 '바탕 화면' 같은 한글 경로를 제대로 처리하지 못하기 때문.
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import pymeshfix
import trimesh


# ---------------------------------------------------------------- 입출력

def load_mesh(path):
    """PLY/OBJ/STL 등을 읽어 (vertices, faces, colors|None) numpy 배열로 반환."""
    tm = trimesh.load(path, force="mesh", process=False)
    colors = None
    if tm.visual.kind == "vertex":
        colors = np.asarray(tm.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
    return np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64), colors


def save_mesh(path, vertices, faces, colors=None):
    """확장자(.stl/.ply/.obj/.glb)에 맞춰 저장. 색상은 PLY/OBJ/GLB에만 남는다."""
    kwargs = {}
    if colors is not None:
        kwargs["vertex_colors"] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    trimesh.Trimesh(vertices, faces, process=False, **kwargs).export(path)


def to_o3d(vertices, faces, colors=None):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                  o3d.utility.Vector3iVector(faces.astype(np.int32)))
    if colors is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(colors)
    m.compute_vertex_normals()
    return m


def from_o3d(mesh):
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    return np.asarray(mesh.vertices).copy(), np.asarray(mesh.triangles).astype(np.int64), \
        (colors.copy() if colors is not None else None)


# ---------------------------------------------------------------- 바닥 정렬

def floor_alignment(vertices, distance=0.008):
    """가장 큰 평면(책상/바닥)을 찾아 z=0, 법선을 +Z로 보내는 4x4 변환을 구한다.

    평면을 못 찾으면 카메라 좌표(y 아래, z 앞)를 z-up으로만 돌린다.
    반환: (T, found_plane)
    """
    pts = vertices
    if len(pts) > 200_000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 200_000, replace=False)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # 첫 프레임 카메라 기준 '위' 방향 (카메라 y축이 아래). 카메라를 대략 똑바로 들고
    # 시작했다고 보고, 이 방향과 45도 이내로 누운 평면만 바닥으로 인정한다.
    # (얼굴/몸/벽처럼 서 있는 면을 바닥으로 오인하지 않도록)
    cam_up = np.array([0.0, -1.0, 0.0])
    found = False
    rest = pcd
    for _ in range(3):  # 큰 평면부터 최대 3개 후보 검사
        if len(rest.points) < 100:
            break
        (a, b, c, d), inliers = rest.segment_plane(distance, 3, 1000)
        if len(inliers) < 0.10 * len(pts):
            break
        n = np.array([a, b, c])
        norm = np.linalg.norm(n)
        n, d = n / norm, d / norm
        if abs(n @ cam_up) > np.cos(np.deg2rad(45)):
            found = True
            if n @ cam_up < 0:
                n, d = -n, -d
            break
        rest = rest.select_by_index(inliers, invert=True)

    if found:
        R = _rotation_between(n, np.array([0.0, 0.0, 1.0]))
        T = np.eye(4)
        T[:3, :3] = R
        T[2, 3] = d  # 평면 위의 점 p: n·p = -d  →  회전 후 z = -d, 그래서 +d로 0에 맞춤
        above = (pts @ R.T)[:, 2] + d > 2 * distance
        center_src = pts[above] if above.sum() > 50 else pts
    else:
        # 카메라 좌표(x 오른쪽, y 아래, z 앞) → (x, z 앞을 y로, -y를 z로)
        T = np.eye(4)
        T[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
        center_src = pts

    # 물체가 원점 근처에 오도록 xy 이동
    c = (center_src @ T[:3, :3].T + T[:3, 3]).mean(axis=0)
    T[0, 3] -= c[0]
    T[1, 3] -= c[1]
    if not found:
        T[2, 3] -= (pts @ T[:3, :3].T)[:, 2].min()
    return T, found


def _rotation_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180도: a에 수직인 아무 축으로
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def transform(vertices, T):
    return vertices @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------- 박스 자르기

@dataclass
class Box:
    """z축 회전만 허용하는 자르기 박스 (단위: m, 정렬된 장면 좌표)."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    yaw_deg: float = 0.0

    def rotation(self):
        t = np.deg2rad(self.yaw_deg)
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    def to_local(self, pts):
        """장면 좌표 → 박스 좌표 (박스는 원점 기준으로 yaw 회전)."""
        return pts @ self.rotation()  # == (R^T p^T)^T

    def corners(self):
        xs, ys, zs = (self.xmin, self.xmax), (self.ymin, self.ymax), (self.zmin, self.zmax)
        local = np.array([[x, y, z] for x in xs for y in ys for z in zs])
        return local @ self.rotation().T

    def lineset(self, color=(1.0, 0.55, 0.0)):
        edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3],
                 [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(self.corners()),
                                  o3d.utility.Vector2iVector(edges))
        ls.paint_uniform_color(color)
        return ls


def suggest_box(vertices, faces, floor_found, pad=0.01):
    """가장 큰 연결 덩어리를 감싸는 박스를 제안한다.
    바닥이 있으면 바닥 위(z>8mm)만 보고, 박스 아래면은 바닥 바로 위로 둔다."""
    sel_tri = (vertices[faces][:, :, 2] > 0.008).all(axis=1) if floor_found \
        else np.ones(len(faces), bool)
    if sel_tri.sum() > 50:
        m = to_o3d(vertices, faces[sel_tri])
        labels, counts, _ = m.cluster_connected_triangles()
        labels = np.asarray(labels)
        sel = faces[sel_tri][labels == np.asarray(counts).argmax()]
        pts = vertices[np.unique(sel)]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        zmin = 0.003 if floor_found else lo[2] - pad
        return Box(lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, zmin, hi[2] + pad)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    return Box(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def inside_mask(vertices, box):
    p = box.to_local(vertices)
    return ((p[:, 0] >= box.xmin) & (p[:, 0] <= box.xmax) &
            (p[:, 1] >= box.ymin) & (p[:, 1] <= box.ymax) &
            (p[:, 2] >= box.zmin) & (p[:, 2] <= box.zmax))


def split_by_box(vertices, faces, box):
    """미리보기용 빠른 분할: 세 꼭짓점이 모두 박스 안이면 inside."""
    m = inside_mask(vertices, box)
    tri_in = m[faces].all(axis=1)
    return faces[tri_in], faces[~tri_in]


def crop_exact(vertices, faces, colors, box):
    """박스의 6개 면으로 메시를 정확히 잘라낸다 (단면이 평평하게 남음)."""
    R = box.rotation()
    local = vertices @ R
    kwargs = {}
    if colors is not None:
        kwargs["vertex_colors"] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    tm = trimesh.Trimesh(local, faces, process=False, **kwargs)
    planes = [([box.xmin, 0, 0], [1, 0, 0]), ([box.xmax, 0, 0], [-1, 0, 0]),
              ([0, box.ymin, 0], [0, 1, 0]), ([0, box.ymax, 0], [0, -1, 0]),
              ([0, 0, box.zmin], [0, 0, 1]), ([0, 0, box.zmax], [0, 0, -1])]
    for origin, normal in planes:
        if len(tm.faces) == 0:
            break
        tm = tm.slice_plane(origin, normal)
    v = np.asarray(tm.vertices) @ R.T
    c = None
    if colors is not None and tm.visual.kind == "vertex" and len(tm.vertices):
        c = np.asarray(tm.visual.vertex_colors)[:, :3] / 255.0
    return v, np.asarray(tm.faces, np.int64), c


# ---------------------------------------------------------------- 프린팅용 정리

@dataclass
class PrintOptions:
    largest_only: bool = True
    smooth_iterations: int = 5
    target_triangles: int = 200_000   # 0이면 줄이지 않음
    make_watertight: bool = True
    scale_to_mm: bool = True


@dataclass
class PrintResult:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None
    watertight: bool
    size_mm: np.ndarray
    volume_cm3: float | None
    log: list


def process_for_print(vertices, faces, colors, box, opts: PrintOptions):
    log = []
    v, f, c = crop_exact(vertices, faces, colors, box)
    if len(f) == 0:
        raise ValueError("박스 안에 메시가 없습니다.")
    log.append(f"자르기: 삼각형 {len(f):,}개")

    mesh = to_o3d(v, f, c)
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    if opts.largest_only:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        if len(counts) > 1:
            mesh.remove_triangles_by_mask(labels != counts.argmax())
            mesh.remove_unreferenced_vertices()
            log.append(f"가장 큰 덩어리만 남김: 조각 {len(counts) - 1}개 제거")

    if opts.smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=opts.smooth_iterations)
        log.append(f"부드럽게: Taubin {opts.smooth_iterations}회")

    if opts.target_triangles and len(mesh.triangles) > opts.target_triangles:
        before = len(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(opts.target_triangles)
        log.append(f"단순화: {before:,} → {len(mesh.triangles):,}")

    v, f, c = from_o3d(mesh)

    if opts.make_watertight:
        mf = pymeshfix.MeshFix(v, f.astype(np.int32))
        mf.repair(joincomp=False, remove_smallest_components=opts.largest_only)
        v, f = np.asarray(mf.points, np.float64), np.asarray(mf.faces, np.int64)
        c = None  # MeshFix는 색을 유지하지 않음 (STL엔 어차피 색이 없음)
        log.append(f"구멍 메우기(MeshFix): 삼각형 {len(f):,}개")
        if len(f) == 0:
            raise ValueError("구멍 메우기 결과가 비었습니다. 박스를 조정해 보세요.")

    # 바닥을 z=0, xy 중심을 원점으로
    lo, hi = v.min(axis=0), v.max(axis=0)
    v = v - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    if opts.scale_to_mm:
        v = v * 1000.0

    tm = trimesh.Trimesh(v, f, process=False)
    watertight = bool(tm.is_watertight)
    unit = 1.0 if opts.scale_to_mm else 1000.0
    size_mm = (v.max(axis=0) - v.min(axis=0)) * unit
    volume = abs(tm.volume) * unit ** 3 / 1000.0 if watertight else None
    log.append("수밀(watertight): " + ("예" if watertight else "아니오"))
    return PrintResult(v, f, c, watertight, size_mm, volume, log)
