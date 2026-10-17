"""스캔 후 정밀 재구성 (오프라인).

실시간 추적은 한 번 틀린 프레임도 그대로 모델에 섞어 버리고, 사람 모드처럼 조명은 고정인데
대상만 도는 경우 밝기 패턴에 끌려 회전을 덜 잡는다. 여기서는 녹화된 프레임으로 처음부터 다시 계산한다.

  1. 키프레임마다 SIFT 특징점 + 깊이 → 3D 특징점 (조명 영향 적음, 회전 방향을 정확히 잡음)
  2. 가까운 키프레임 쌍(1·2·3칸)과 한 바퀴 돌아온 쌍(루프)을 특징점 RANSAC → ICP로 정합
  3. 사람 모드: 회전의자는 한 축으로만 돌므로 다른 축으로 돈 정합은 버림
  4. 포즈 그래프 최적화 → 주변과 안 맞는 프레임은 버리고 다시 최적화
  5. 남은 프레임만 깨끗한 TSDF에 다시 융합
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

reg = o3d.pipelines.registration


@dataclass
class RefineConfig:
    stride: int = 2                 # 키프레임 간격 (프레임)
    voxel_size: float = 0.002       # 정밀 재구성은 2mm (실시간은 4mm). 얼굴 윤곽이 살아남
    depth_max: float = 1.0
    feature_inlier_m: float = 0.012  # 특징점 RANSAC 허용 오차
    min_feature_inliers: int = 12
    icp_dist_m: float = 0.012
    loop_max_angle_deg: float = 30.0  # 루프 후보: 시선 방향 차이가 이 이하인 먼 프레임 쌍
    max_loop_pairs: int = 400
    axis_tolerance_deg: float = 35.0  # 사람 모드: 주 회전축과 이보다 다르면 '불확실' 정합으로 취급
    drop_trans_m: float = 0.012       # 최적화 후 이 이상 안 맞는 프레임은 버림
    drop_rot_deg: float = 3.0
    # 사람 모드: 정합(특징점·형태)은 머리만으로. 머리는 모양이 안 변하는 강체지만 어깨·옷은 돌면서 계속
    # 변해서, 함께 맞추면 넓은 어깨 쪽에 끌려가 고개가 조금만 돌아도 얼굴이 두 겹이 된다.
    early_seams: bool = False          # 최적화 전에 특징점만으로 이음매를 '확실'하게 묶기 (한 쌍만 틀려도 전체가 휨 → 기본 끔)
    jump_factor: float = 4.0           # 이웃 프레임 회전이 주변 속도의 이 배수(+6도)를 넘으면 튄 정합으로 봄
    final_seam: bool = True
    seam_ref_frames: int = 10          # 마지막 연결에서 끝 장면과 순서대로 비교할 처음 장면 수 (N)            # 최적화 후 처음 장면과 한 바퀴 뒤 장면을 반복해서 맞춤
    head_only_registration: bool = True
    head_height_m: float = 0.26       # 정수리부터 이 높이까지를 머리로 봄
    weight_threshold: float = 1.5     # 메시 추출 시 최소 관측 가중치 (낮을수록 경계가 덜 사라짐)
    frame_reject_far_m: float = 0.005   # 완성 모델과 비교: 이보다 멀리 떨어진 점을 '안 맞음'으로 셈
    frame_reject_ratio: float = 0.35    # 안 맞는 점 비율이 이보다 높고(그리고 중앙값의 3배 이상) 프레임만 버림
    model_refine_iters: int = 0       # 융합된 모델에 각 프레임을 다시 맞춰 재융합하는 횟수 (두 겹 표면 감소)
    model_refine_dist_m: float = 0.006


@dataclass
class _Key:
    idx: int
    depth: np.ndarray
    color: np.ndarray
    pass_id: int = 0
    pts: np.ndarray = None      # 특징점 3D (N,3)
    desc: np.ndarray = None
    pcd: o3d.geometry.PointCloud = None


@dataclass
class RefineReport:
    keyframes: int = 0
    used: int = 0
    dropped: list = field(default_factory=list)
    edges: int = 0
    loops: int = 0
    rejected_axis: int = 0
    removed_edges: int = 0
    poses: dict = field(default_factory=dict)   # 프레임 번호 → 최적화된 자세
    log: list = field(default_factory=list)


# ---------------------------------------------------------------- 유틸

def clean_depth(depth, scale, jump_m=0.02, erode_px=2):
    """실루엣 경계의 튀는 깊이(flying pixel)를 제거. 경계에서 생기는 줄무늬/잔상 방지."""
    d = depth.astype(np.float32) / scale
    valid = d > 0
    pad = np.pad(d, 1, mode="edge")
    jump = np.zeros_like(d, bool)
    for dy, dx in ((0, 1), (1, 0), (2, 1), (1, 2)):
        nb = pad[dy:dy + d.shape[0], dx:dx + d.shape[1]]
        jump |= (np.abs(nb - d) > jump_m) & (nb > 0)
    keep = valid & ~jump
    if erode_px:
        keep = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode_px) > 0
    out = depth.copy()
    out[~keep] = 0
    return out


def _motion(T):
    m = np.linalg.norm(T[:3, 3])
    deg = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    return m, deg


def _axis_angle(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-6:
        return np.zeros(3), 0.0
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return ax / (np.linalg.norm(ax) + 1e-12), np.degrees(ang)


def _interp_se3(E, t):
    """변환 E를 t(0~1)만큼만 적용한 변환 (회전은 구면 보간, 이동은 선형)."""
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(E[:3, :3]).as_rotvec()
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(rv * t).as_matrix()
    out[:3, 3] = E[:3, 3] * t
    return out


def _kabsch(A, B):
    """B ≈ R A + t 인 강체 변환."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def _ransac(A, B, thr, iters=400, rng=np.random.default_rng(0)):
    n = len(A)
    if n < 3:
        return None, np.zeros(n, bool)
    best = np.zeros(n, bool)
    for _ in range(iters):
        s = rng.choice(n, 3, replace=False)
        T = _kabsch(A[s], B[s])
        err = np.linalg.norm(A @ T[:3, :3].T + T[:3, 3] - B, axis=1)
        inl = err < thr
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        return None, best
    T = _kabsch(A[best], B[best])
    err = np.linalg.norm(A @ T[:3, :3].T + T[:3, 3] - B, axis=1)
    best = err < thr
    return _kabsch(A[best], B[best]) if best.sum() >= 3 else None, best


# ---------------------------------------------------------------- 정밀 재구성

class Refiner:
    def __init__(self, K, depth_scale_factor, cfg: RefineConfig, subject_mode=False, log=print):
        self.K = np.asarray(K, float)
        self.scale = depth_scale_factor
        self.cfg = cfg
        self.subject = subject_mode
        self.log = log
        self.sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.02)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.intr_o3d = o3d.camera.PinholeCameraIntrinsic(
            0, 0, self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])

    # ---------------- 키프레임 준비

    def _prepare(self, k: _Key):
        c = self.cfg
        k.depth = clean_depth(k.depth, self.scale)
        z = k.depth.astype(np.float32) / self.scale
        z[z > c.depth_max] = 0
        k.depth[z == 0] = 0
        # 정합에 쓸 영역 (사람 모드면 머리만). 융합에는 k.depth 전체를 그대로 쓴다.
        zr = self._head_only(z) if (self.subject and c.head_only_registration) else z
        # 조명 차이를 줄이려 CLAHE로 국소 대비를 정규화한 뒤 특징점 추출
        gray = self.clahe.apply(cv2.cvtColor(k.color, cv2.COLOR_RGB2GRAY))
        mask = (cv2.erode((zr > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) * 255)
        kps, desc = self.sift.detectAndCompute(gray, mask)
        if desc is None or len(kps) == 0:
            k.pts, k.desc = np.zeros((0, 3)), np.zeros((0, 128), np.float32)
        else:
            uv = np.array([kp.pt for kp in kps])
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            zz = z[v, u]
            ok = zz > 0
            fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            k.pts = np.stack([(uv[ok, 0] - cx) * zz[ok] / fx, (uv[ok, 1] - cy) * zz[ok] / fy, zz[ok]], 1)
            k.desc = desc[ok]
        # ICP용 점군
        h, w = z.shape
        vv, uu = np.nonzero(zr > 0)
        zz = zr[vv, uu]
        P = np.stack([(uu - self.K[0, 2]) * zz / self.K[0, 0], (vv - self.K[1, 2]) * zz / self.K[1, 1], zz], 1)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(c.voxel_size)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.zeros(3))
        k.pcd = pcd

    def _head_only(self, z):
        """화면 위쪽에서 처음 대상이 나타나는 줄(정수리)부터 head_height_m 만큼만 남긴 깊이."""
        rows = np.flatnonzero((z > 0).sum(axis=1) > 8)
        if len(rows) == 0:
            return z
        top = rows[0]
        zt = z[top:top + 40]
        zmed = float(np.median(zt[zt > 0])) if (zt > 0).any() else 0.5
        h_px = int(self.cfg.head_height_m * self.K[1, 1] / max(zmed, 0.2))
        out = z.copy()
        out[top + h_px:] = 0
        return out

    # ---------------- 쌍 정합

    def _register(self, a: _Key, b: _Key, init=None, trust=(0.05, 10.0)):
        """a → b 변환 (p_b = T p_a). 반환: (T, info, 설명) 또는 (None, None, 설명).
        init: 예상 변환 하나 또는 여러 후보의 리스트(예: 직전처럼 계속 돎 / 반대로 돎 / 멈춤 / 실시간 추적).
        특징점 결과가 어느 후보와도 trust(m, 도) 이상 다르면 특징점을 무시하고(반복 무늬 등으로 엉뚱한 곳을
        맞춘 경우), 특징점이 없으면 후보마다 형태 정합(ICP)을 해서 가장 잘 맞는 것을 쓴다."""
        inits = [] if init is None else (list(init) if isinstance(init, (list, tuple)) else [init])
        if len(inits) > 1:
            best = (None, None, "no candidate", -1.0)
            for cand in inits:
                T, info, why = self._register(a, b, cand, trust)
                if T is not None:
                    fit = reg.evaluate_registration(a.pcd, b.pcd, self.cfg.icp_dist_m, T).fitness
                    if fit > best[3]:
                        best = (T, info, why, fit)
                    if "features" in why and not why.startswith("features 0"):
                        if int(why.split()[1].rstrip(",")) >= self.cfg.min_feature_inliers:
                            return T, info, why     # 특징점으로 확실히 맞음 → 더 볼 필요 없음
            return best[:3]
        init = inits[0] if inits else None
        c = self.cfg
        T_feat, n_inl = None, 0
        if len(a.desc) >= 8 and len(b.desc) >= 8:
            pairs = self.matcher.knnMatch(a.desc, b.desc, k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.8 * n.distance]
            if len(good) >= c.min_feature_inliers:
                A = a.pts[[m.queryIdx for m in good]]
                B = b.pts[[m.trainIdx for m in good]]
                T_feat, inl = _ransac(A, B, c.feature_inlier_m)
                n_inl = int(inl.sum())
                if n_inl < c.min_feature_inliers:
                    T_feat = None
                elif init is not None:
                    m, deg = _motion(np.linalg.inv(init) @ T_feat)
                    if m > trust[0] or deg > trust[1]:
                        T_feat = None
        start = T_feat if T_feat is not None else init
        if start is None:
            return None, None, f"features {n_inl}"
        est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=c.icp_dist_m))
        icp = reg.registration_icp(a.pcd, b.pcd, c.icp_dist_m, start, est,
                                   reg.ICPConvergenceCriteria(max_iteration=30))
        T = icp.transformation
        if T_feat is not None:
            # ICP가 형태만 보고 미끄러져(머리처럼 둥근 대상) 특징점과 어긋나면 특징점 결과를 유지
            A = a.pts[[m.queryIdx for m in good]][inl]
            B = b.pts[[m.trainIdx for m in good]][inl]
            err = np.linalg.norm(A @ T[:3, :3].T + T[:3, 3] - B, axis=1)
            if (err < c.feature_inlier_m).mean() < 0.7:
                T = T_feat
        ev = reg.evaluate_registration(a.pcd, b.pcd, c.icp_dist_m, T)
        if ev.fitness < 0.3 or (T_feat is None and ev.fitness < 0.5):
            return None, None, f"features {n_inl}, fitness {ev.fitness:.2f}"
        info = reg.get_information_matrix_from_point_clouds(a.pcd, b.pcd, c.icp_dist_m, T)
        return T, info, f"features {n_inl}, fitness {ev.fitness:.2f}"

    # ---------------- 메인

    def run(self, frames, live_poses=None, passes=None):
        """frames: [(depth uint16, color uint8 RGB)] 또는 frames(i)를 주는 (개수, 함수) 튜플.
        live_poses: 실시간 추적 자세(없어도 됨). passes: 프레임별 패스 번호(카메라를 옮겨 다시 찍은 구간)."""
        c, rep = self.cfg, RefineReport()
        if isinstance(frames, tuple):
            count, get = frames
        else:
            count, get = len(frames), frames.__getitem__
        passes = list(passes) if passes is not None else [0] * count
        keys = []
        for i in range(0, count, c.stride):
            d, col = get(i)
            keys.append(_Key(i, d.copy(), col, passes[i]))
        rep.keyframes = len(keys)
        self.log(f"정밀 재구성: 키프레임 {len(keys)}개 준비 중...")
        for k in keys:
            self._prepare(k)
        keys = [k for k in keys if len(k.pcd.points) > 500]
        n = len(keys)
        if n < 3:
            raise ValueError("쓸 수 있는 프레임이 너무 적습니다.")

        # 1) 가까운 쌍 정합 (1·2·3칸 떨어진 키프레임)
        self.log("가까운 프레임끼리 특징점 정합 중...")
        edges = {}   # (i, j) -> (T_ij: p_j = T p_i, info, loop)
        last_step = None   # 직전에 성공한 1칸 움직임 (특징점이 부족한 구간에서 예측값으로 사용)
        for i in range(n):
            if i and keys[i].pass_id != keys[i - 1].pass_id:
                last_step = None
            for gap in (1, 2, 3):
                j = i + gap
                if j >= n:
                    break
                if keys[i].pass_id != keys[j].pass_id:
                    continue
                # 특징점이 부족하면(뒤통수 머리카락 등) 이 예측 위치에서 형태 정합(ICP)으로 대신 잇는다.
                # 회전의자는 거의 일정한 속도로 도므로 직전 속도가 실시간 추적보다 믿을 만하다.
                # 시작 위치 후보: 직전처럼 같은 방향으로 계속 / 반대 방향으로 돌아섬 / 멈춤 / 실시간 추적.
                # (한 방향만 가정하면 회전 방향을 바꾼 순간 정합을 놓친다)
                init = []
                if last_step is not None:
                    init += [np.linalg.matrix_power(last_step, gap),
                             np.linalg.matrix_power(np.linalg.inv(last_step), gap)]
                if live_poses is not None:
                    init.append(np.linalg.inv(live_poses[keys[j].idx]) @ live_poses[keys[i].idx])
                init.append(np.eye(4))
                T, info, _ = self._register(keys[i], keys[j], init)
                if T is not None:
                    edges[(i, j)] = (T, info, False)
                    if gap == 1:
                        last_step = T

        # 2) 사람 모드: 주 회전축과 크게 다른 축으로 돈 정합은 '불확실'로 표시 (지우지 않음).
        #    작은 회전(4도 미만)은 축 방향 자체가 부정확하므로 판단하지 않는다.
        for pid in (sorted(set(k.pass_id for k in keys)) if self.subject else []):
            axes = []
            for (i, j), (T, _, _) in edges.items():
                if keys[i].pass_id != pid:
                    continue
                ax, deg = _axis_angle(T[:3, :3])
                if deg > 1.0:
                    axes.append(ax * deg)
            if axes:
                axes = np.array(axes)
                ref = axes[np.argmax(np.linalg.norm(axes, axis=1))]
                axes = axes * np.sign(axes @ ref)[:, None]
                main = np.median(axes, axis=0)
                main /= np.linalg.norm(main)
                for key in list(edges):
                    if keys[key[0]].pass_id != pid:
                        continue
                    T, info, _ = edges[key]
                    ax, deg = _axis_angle(T[:3, :3])
                    if deg > 4.0 and np.degrees(np.arccos(min(1.0, abs(ax @ main)))) > c.axis_tolerance_deg:
                        edges[key] = (T, info, True)
                        rep.rejected_axis += 1
                self.log(f"패스 {pid + 1} 회전축 {np.round(main, 2)}: 축이 어긋난 정합 누적 {rep.rejected_axis}개를 불확실로 표시")

        # 2-1) 갑자기 튄 정합 제거: 회전의자는 거의 일정한 속도로 돌기 때문에, 주변 프레임들보다
        #      훨씬 크게 돈 정합(예: 다른 곳들은 3도씩인데 혼자 40도)은 엉뚱한 곳에 붙은 것이다.
        jumps = self._remove_jumps(keys, edges)
        if jumps:
            self.log(f"  주변 회전 속도와 안 맞게 튄 정합 {jumps}개 제외")

        # 2-2) 정합이 하나도 없는 이웃 사이는 실시간 추적 자세로 '약하게' 이어 둔다.
        #      (이게 없으면 그 구간에서 그래프가 끊겨 뒤쪽 프레임이 통째로 버려진다)
        if live_poses is not None:
            weak = 0
            for i in range(n - 1):
                j = i + 1
                if keys[i].pass_id != keys[j].pass_id:
                    continue
                if not any((i, j + g) in edges or (i - g, j) in edges for g in (0, 1, 2)):
                    T = np.linalg.inv(live_poses[keys[j].idx]) @ live_poses[keys[i].idx]
                    edges[(i, j)] = (T, np.eye(6) * 50.0, False)
                    weak += 1
            if weak:
                self.log(f"  정합이 끊긴 {weak}곳을 실시간 추적 자세로 약하게 연결")

        # 3) 대략적 자세 (연결된 체인 따라 누적) → 루프 후보 찾기
        poses = self._chain_poses(n, edges)
        self.log("한 바퀴 돌아온 프레임(루프) 찾는 중...")
        cands = []
        for i in range(n):
            for j in range(i + 6, n):
                if poses[i] is None or poses[j] is None or keys[i].pass_id != keys[j].pass_id:
                    continue
                rel = np.linalg.inv(poses[j]) @ poses[i]
                m, deg = _motion(rel)
                if deg < c.loop_max_angle_deg and m < 0.3:
                    cands.append((deg, i, j))
        cands.sort()
        step = max(1, len(cands) // c.max_loop_pairs)
        for _, i, j in cands[::step][:c.max_loop_pairs]:
            init = np.linalg.inv(poses[j]) @ poses[i]
            T, info, _ = self._register(keys[i], keys[j], init, trust=(0.15, 25.0))
            if T is not None:
                m, deg = _motion(np.linalg.inv(init) @ T)
                if m < 0.15 and deg < 25:
                    edges[(i, j)] = (T, info, True)
                    rep.loops += 1
        self.log(f"루프 정합 {rep.loops}개 (후보 {len(cands)}개)")

        # 3-1) 이음매 닫기: 한 바퀴 돌아 처음 본 곳을 다시 본 프레임을 특징점으로 직접 맞추고,
        #      그 어긋남을 중간 프레임들에 고르게 나눠(보간) 초기 자세를 바로잡는다.
        if self.cfg.early_seams:
            poses = self._close_seams(keys, edges, poses, rep)

        # 3-2) 패스끼리 이어 붙이기: 자세를 모르므로 특징점이 많이 겹치는 쌍부터 정합
        pids = sorted(set(k.pass_id for k in keys))
        if len(pids) > 1:
            self.log("패스끼리 이어 붙이는 중...")
            for a_i, pa in enumerate(pids):
                for pb in pids[a_i + 1:]:
                    A = [i for i in range(n) if keys[i].pass_id == pa][::2]
                    B = [j for j in range(n) if keys[j].pass_id == pb][::2]
                    scored = []
                    for i in A:
                        for j in B:
                            if len(keys[i].desc) < 8 or len(keys[j].desc) < 8:
                                continue
                            pairs = self.matcher.knnMatch(keys[i].desc, keys[j].desc, k=2)
                            g = sum(1 for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)
                            scored.append((g, i, j))
                    scored.sort(reverse=True)
                    links = []
                    for g, i, j in scored[:80]:
                        T, info, _ = self._register(keys[i], keys[j], None)
                        if T is not None:
                            links.append((i, j, T, info))
                    linked = self._link_passes(keys, edges, pa, pb, links)
                    rep.loops += linked
                    self.log(f"  패스 {pa + 1} ↔ {pb + 1}: 연결 {linked}개 (특징점 직접 연결 {len(links)}개)")
            poses = self._chain_poses(n, edges)

        # 4) 포즈 그래프 최적화 → 최적화 결과와 가장 안 맞는 '정합(연결)'부터 조금씩 지우고 반복.
        #    틀린 연결 하나가 그래프를 휘게 만들어 멀쩡한 프레임까지 버려지는 것을 막는다.
        alive = set(range(n))
        removed = 0
        for it in range(8):
            P = self._optimize(n, edges, alive, poses)
            res = []
            for key, (T, _, _) in edges.items():
                i, j = key
                if i in alive and j in alive:
                    m, deg = _motion(np.linalg.inv(np.linalg.inv(P[j]) @ P[i]) @ T)
                    res.append((max(m / c.drop_trans_m, deg / c.drop_rot_deg), key))
            # 허용 오차의 2배 넘게 벗어난 것만 '확실히 틀린' 정합으로 본다
            bad = sorted((r for r in res if r[0] > 2.0), reverse=True)
            if not bad:
                break
            n_removed = 0
            for _, key in bad[:max(1, len(res) // 20)]:
                if edges[key][2] == "seam":
                    continue    # 이음매 연결은 기준점이므로 지우지 않음
                if not self._connected_without(edges, alive, key):
                    continue    # 이걸 지우면 그래프가 끊김 → 남겨 둔다 (프레임을 잃지 않게)
                del edges[key]
                removed += 1
                n_removed += 1
            if n_removed == 0:
                break
        rep.removed_edges = removed
        before = set(range(n))
        alive = self._largest_component(edges, alive)
        P = self._optimize(n, edges, alive, poses)
        rep.dropped = [keys[b].idx for b in sorted(before - alive)]
        self.log(f"  안 맞는 정합 {removed}개 제거, 연결이 끊겨 빠진 프레임 {len(rep.dropped)}개")
        rep.used = len(alive)
        rep.edges = sum(1 for (i, j) in edges if i in alive and j in alive)
        rep.poses = {keys[i].idx: P[i] for i in alive}

        # 4-2) 마지막 이음매 맞추기: 최적화가 끝난 뒤에(다시 흐트러지지 않게) 처음 장면과 한 바퀴 뒤 장면을
        #      한 프레임씩 비교해 가장 자연스럽게 맞는 프레임을 찾고, 어긋남을 회전각 비율로 나눠 보정. 반복.
        P = self._final_seam(keys, P, alive)
        rep.poses = {keys[i].idx: P[i] for i in alive}

        # 5) 다시 융합
        self.log(f"남은 키프레임 {len(alive)}/{n}개로 다시 융합 중...")
        ids = sorted(alive)
        v, f, col = self._integrate([keys[i] for i in ids], [P[i] for i in ids])
        # 5-2) 완성된 모델과 안 맞는 프레임 제거 후 재융합
        #      (예: SPACE/ENTER 누르느라 팔·몸이 움직인 시작·끝 구간, 사람이 자세를 바꾼 순간)
        far = self._frame_disagreement(keys, ids, P, v, f)
        med = float(np.median(list(far.values())))
        bad = [i for i in ids if far[i] > max(c.frame_reject_ratio, 3.0 * med)]
        if bad and len(bad) < 0.5 * len(ids):
            ids = [i for i in ids if i not in bad]
            rep.dropped += [keys[i].idx for i in bad]
            v, f, col = self._integrate([keys[i] for i in ids], [P[i] for i in ids])
            self.log(f"  완성 모델과 안 맞는 프레임 {len(bad)}개 제외 (안 맞는 점 비율 중앙값 {med:.0%})")
        rep.used = len(ids)
        # 6) 모델 기준 미세 정렬: 이웃 프레임끼리보다 전체 모델과 맞추는 게 안정적이다.
        for it in range(c.model_refine_iters):
            moved = self._refine_to_model(keys, ids, P, v, f)
            v, f, col = self._integrate([keys[i] for i in ids], [P[i] for i in ids])
            self.log(f"  모델 기준 미세 정렬 {it + 1}: 평균 보정 {moved * 1000:.2f}mm")
        rep.poses = {keys[i].idx: P[i] for i in ids}
        rep.log.append(f"키프레임 {n}개 중 {rep.used}개 사용, 정합 {rep.edges}개 (루프 {rep.loops}), "
                       f"축 어긋남 제외 {rep.rejected_axis}, 제거한 정합 {rep.removed_edges}, 빠진 프레임 {len(rep.dropped)}")
        self.log(rep.log[-1])
        return v, f, col, rep

    def _link_passes(self, keys, edges, pa, pb, links):
        """두 패스 사이 연결. 특징점으로 직접 맞춘 몇 개의 연결에서 패스 간 상대 위치를 합의로 정하고
        (틀린 연결 하나에 끌려가지 않게), 그 위치로 예측한 모든 쌍을 다시 정합해 연결을 늘린다."""
        n = len(keys)
        A = [i for i in range(n) if keys[i].pass_id == pa]
        B = [j for j in range(n) if keys[j].pass_id == pb]
        PA = self._chain_poses(n, edges, set(A))
        PB = self._chain_poses(n, edges, set(B))
        offsets = []   # 패스 b 좌표 → 패스 a 좌표
        for i, j, T, _ in links:
            if PA[i] is not None and PB[j] is not None:
                offsets.append(PA[i] @ np.linalg.inv(T) @ np.linalg.inv(PB[j]))
        O, best_support = None, 0
        if offsets:
            support = [sum(1 for O2 in offsets
                           if (lambda m, d: m < 0.03 and d < 5)(*_motion(np.linalg.inv(O1) @ O2)))
                       for O1 in offsets]
            O, best_support = offsets[int(np.argmax(support))], max(support)
        # 직접 연결이 서로 3개 이상 합의하지 못하면 3D 형태로 두 패스 모델 전체를 맞춘다
        # (반복 무늬·조명 차이로 특징점이 엉뚱한 쌍을 맞추는 경우 대비)
        cand = [O] if O is not None and best_support >= 3 else []
        G = self._global_pass_alignment(keys, A, B, PA, PB)
        if G is not None:
            cand.append(G)
        if not cand:
            return 0
        model_a = self._pass_cloud(keys, A, PA)
        model_b = self._pass_cloud(keys, B, PB)
        scores = [reg.evaluate_registration(model_b, model_a, self.cfg.icp_dist_m, C).fitness for C in cand]
        O = cand[int(np.argmax(scores))]
        self.log(f"    패스 상대 위치: 겹침 {max(scores):.2f} "
                 f"({'특징점 합의' if O is cand[0] and best_support >= 3 else '3D 형태 정합'})")
        count = 0
        for i, j, T, info in links:   # 합의와 맞는 직접 연결만 채택
            if PA[i] is not None and PB[j] is not None:
                m, d = _motion(np.linalg.inv(PA[i] @ np.linalg.inv(T) @ np.linalg.inv(PB[j])) @ O)
                if m < 0.03 and d < 5:
                    edges[(i, j)] = (T, info, True)
                    count += 1
        # 예측 위치로 보는 방향이 비슷한 쌍을 골라 추가 정합
        cands = []
        for i in A[::2]:
            for j in B[::2]:
                if PA[i] is None or PB[j] is None or (i, j) in edges:
                    continue
                pred = np.linalg.inv(O @ PB[j]) @ PA[i]
                m, d = _motion(pred)
                if d < 50 and m < 1.0:   # 카메라 높이를 바꾼 패스는 위치 차이가 클 수 있어 각도로만 거름
                    cands.append((d, i, j, pred))
        cands.sort(key=lambda x: x[0])
        for _, i, j, pred in cands[:200]:
            T, info, _ = self._register(keys[i], keys[j], pred, trust=(0.04, 8.0))
            if T is None:
                continue
            m, d = _motion(np.linalg.inv(pred) @ T)
            if m < 0.04 and d < 8:
                edges[(i, j)] = (T, info, True)
                count += 1
        return count

    def _pass_cloud(self, keys, ids, poses, voxel=0.005):
        pcd = o3d.geometry.PointCloud()
        for k in ids:
            if poses[k] is not None:
                pcd += o3d.geometry.PointCloud(keys[k].pcd).transform(poses[k])
        pcd = pcd.voxel_down_sample(voxel)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        return pcd

    def _global_pass_alignment(self, keys, A, B, PA, PB):
        """패스 b 모델 → 패스 a 모델 변환을 형태 기술자(FPFH) + RANSAC으로 찾고 ICP로 다듬는다."""
        c = self.cfg
        va, vb = self._pass_cloud(keys, A, PA, 0.008), self._pass_cloud(keys, B, PB, 0.008)
        if len(va.points) < 100 or len(vb.points) < 100:
            return None
        param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
        fa = reg.compute_fpfh_feature(va, param)
        fb = reg.compute_fpfh_feature(vb, param)
        res = reg.registration_ransac_based_on_feature_matching(
            vb, va, fb, fa, True, 0.015, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(0.015)],
            reg.RANSACConvergenceCriteria(200000, 0.999))
        fine_a, fine_b = self._pass_cloud(keys, A, PA), self._pass_cloud(keys, B, PB)
        icp = reg.registration_icp(fine_b, fine_a, c.icp_dist_m, res.transformation,
                                   reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=c.icp_dist_m)),
                                   reg.ICPConvergenceCriteria(max_iteration=50))
        return icp.transformation if icp.fitness > 0.2 else None


    def _remove_jumps(self, keys, edges):
        c = self.cfg
        n = len(keys)
        step = {}
        for i in range(n - 1):
            e = edges.get((i, i + 1))
            if e is not None and e[2] is False:
                step[i] = _motion(e[0])[1]
        if len(step) < 10:
            return 0
        idx = np.array(sorted(step))
        val = np.array([step[i] for i in idx])
        removed = 0
        for key in list(edges):
            i, j = key
            T, _, loop = edges[key]
            if loop is not False or j - i > 3:
                continue
            near = val[np.abs(idx - i) <= 10]
            typical = float(np.median(near)) * (j - i)
            deg = _motion(T)[1]
            if deg > c.jump_factor * typical + 6.0:
                del edges[key]
                removed += 1
        return removed

    # ---------------- 이음매 닫기

    def _cumulative_angles(self, keys, edges, ids):
        """연속 정합(1칸)을 따라 주 회전축 기준 누적 회전각(도)."""
        ang = {ids[0]: 0.0}
        axis = None
        for a, b in zip(ids[:-1], ids[1:]):
            e = edges.get((a, b))
            if e is None:
                ang[b] = ang[a]
                continue
            ax, deg = _axis_angle(e[0][:3, :3])
            if axis is None and deg > 2:
                axis = ax
            sgn = 1.0 if axis is None else float(np.sign(ax @ axis) or 1.0)
            ang[b] = ang[a] + sgn * deg
        return ang

    def _close_seams(self, keys, edges, poses, rep):
        c = self.cfg
        n = len(keys)
        # 이음매 계산은 순수한 연속 정합(루프 제외)으로 쌓은 자세 기준: 어긋남이 끝에 그대로 모여 있어야
        # 그걸 재서 고르게 나눌 수 있다
        odo = self._chain_poses(n, {k: v for k, v in edges.items() if v[2] is False})
        poses = [o if o is not None else p for o, p in zip(odo, poses)]
        new_poses = list(poses)
        for pid in sorted(set(k.pass_id for k in keys)):
            ids = [i for i in range(n) if keys[i].pass_id == pid and poses[i] is not None]
            if len(ids) < 20:
                continue
            ang = self._cumulative_angles(keys, edges, ids)
            span = max(abs(ang[i]) for i in ids)
            if span < 270:
                self.log(f"  패스 {pid + 1}: 회전 {span:.0f}도라 이음매 없음 (한 바퀴 미만)")
                continue
            # 처음 구간 프레임 vs 대략 한 바퀴(360±70도) 뒤 프레임
            start = ids[:max(5, len(ids) // 8)]
            cand = []
            for i in start[::2]:
                for j in ids:
                    d = abs(ang[j] - ang[i])
                    if 290 <= d <= 430 and len(keys[i].desc) >= 8 and len(keys[j].desc) >= 8:
                        pairs = self.matcher.knnMatch(keys[i].desc, keys[j].desc, k=2)
                        g = sum(1 for q in pairs if len(q) == 2 and q[0].distance < 0.75 * q[1].distance)
                        cand.append((g, i, j))
            cand.sort(reverse=True)
            found = []
            for g, i, j in cand[:40]:
                base = np.linalg.inv(poses[j]) @ poses[i]       # 누적 자세로 본 예상 변환 (오차 포함)
                T, info, why = self._register(keys[i], keys[j], self._small_perturbations(base, keys[i]),
                                              trust=(0.25, 60.0))
                if T is None:
                    T, info, why = self._register(keys[i], keys[j], None)
                if T is None:
                    continue
                n_feat = int(why.split()[1].rstrip(",")) if why.startswith("features") else 0
                if n_feat < c.min_feature_inliers:
                    continue
                # 이 쌍이 말하는 '끝 자세 보정량' (월드 기준)
                E = (poses[i] @ np.linalg.inv(T)) @ np.linalg.inv(poses[j])
                found.append((i, j, T, info, E))
            if not found:
                self.log(f"  패스 {pid + 1}: 이음매 쌍을 찾지 못했습니다 (특징점 부족)")
                continue
            # 합의: 가장 많은 쌍이 동의하는 보정량
            support = [sum(1 for f2 in found if (lambda m, d: m < 0.02 and d < 3)(
                *_motion(np.linalg.inv(f1[4]) @ f2[4]))) for f1 in found]
            best = found[int(np.argmax(support))]
            agree = [f for f in found if (lambda m, d: m < 0.02 and d < 3)(*_motion(np.linalg.inv(best[4]) @ f[4]))]
            for i, j, T, info, _ in agree:
                edges[(i, j)] = (T, info, "seam")
            m, deg = _motion(best[4])
            # 보간: i~j 사이 프레임에 보정량을 비율만큼, j 이후는 전부
            i0, j0 = best[0], best[1]
            for k in ids:
                if k <= i0:
                    continue
                t = 1.0 if k >= j0 else (k - i0) / (j0 - i0)
                new_poses[k] = _interp_se3(best[4], t) @ poses[k]
            rep.loops += len(agree)
            self.log(f"  패스 {pid + 1}: 이음매 {len(agree)}쌍으로 닫음 (어긋남 {m * 1000:.1f}mm, {deg:.1f}도를 "
                     f"중간 프레임에 나눠 보정, 후보 {len(found)}쌍)")
        return new_poses


    def _angles_from_poses(self, ids, P):
        """최종 자세에서 주 회전축 기준 누적 회전각(도)."""
        ang = {ids[0]: 0.0}
        axis = None
        for a, b in zip(ids[:-1], ids[1:]):
            ax, deg = _axis_angle((np.linalg.inv(P[b]) @ P[a])[:3, :3])
            if axis is None and deg > 1:
                axis = ax
            sgn = 1.0 if axis is None else float(np.sign(ax @ axis) or 1.0)
            ang[b] = ang[a] + sgn * deg
        return ang

    def _world_cloud(self, keys, P, sel, voxel=0.003):
        pc = o3d.geometry.PointCloud()
        for i in sel:
            pc += o3d.geometry.PointCloud(keys[i].pcd).transform(P[i])
        pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
        return pc

    def _icp_multistart(self, src, dst, center):
        """작은 회전(±3·6도)·작은 이동(±5mm) 시작점 여러 개로 ICP → 가장 잘 맞는 결과."""
        starts = [np.eye(4)]
        for deg in (-6, -3, 3, 6):
            for axis in ((0, 1, 0), (1, 0, 0)):
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array(axis, float) * np.deg2rad(deg))
                D = np.eye(4)
                D[:3, :3] = R
                D[:3, 3] = center - R @ center
                starts.append(D)
        for d in ((0.005, 0, 0), (-0.005, 0, 0), (0, 0, 0.005), (0, 0, -0.005)):
            D = np.eye(4)
            D[:3, 3] = d
            starts.append(D)
        est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=0.006))
        best = None
        for S in starts:
            T = S
            for dist in (0.02, 0.008, 0.004):
                r = reg.registration_icp(src, dst, dist, T, est, reg.ICPConvergenceCriteria(max_iteration=30))
                T = r.transformation
            score = (r.fitness, -r.inlier_rmse)
            if best is None or score > best[0]:
                best = (score, T, r.fitness, r.inlier_rmse)
        return best[1], best[2], best[3]

    def _final_seam(self, keys, P, alive):
        """마지막 연결: 처음 장면과 '다시 같은 곳을 본' 프레임을 겉모습(특징점)으로 직접 찾아 맞추고,
        그 어긋남을 돈 정도에 비례해 모든 프레임에 나눠 보정한다. 1mm 밑이 될 때까지 반복.
        (추정 회전각으로 찾으면, 바로 그 각도 오차 때문에 후보를 놓친다)"""
        P = list(P)
        if not self.cfg.final_seam:
            return P
        for pid in sorted(set(keys[i].pass_id for i in alive)):
            ids = sorted(i for i in alive if keys[i].pass_id == pid)
            if len(ids) < 20:
                continue
            n_ref = min(self.cfg.seam_ref_frames, len(ids) // 4)
            ref_ids = ids[:max(n_ref, 3)]
            late = ids[len(ids) // 2:]
            # 1) 처음 N장을 하나씩 순서대로 기준으로 삼아, 뒤쪽 절반의 모든 프레임 중 가장 비슷한 프레임을 찾는다
            found = []          # (기준 r, 끝 프레임 k, 보정 C, 특징점 수, 겹침)
            for r in ref_ids:
                if len(keys[r].desc) < 8:
                    continue
                scored = []
                for k in late:
                    if len(keys[k].desc) < 8:
                        continue
                    pairs = self.matcher.knnMatch(keys[k].desc, keys[r].desc, k=2)
                    g = sum(1 for q in pairs if len(q) == 2 and q[0].distance < 0.75 * q[1].distance)
                    scored.append((g, k))
                scored.sort(reverse=True)
                best_r = None
                for g, k in scored[:5]:
                    T, info, why = self._register(keys[k], keys[r], None)
                    if T is None or not why.startswith("features"):
                        continue
                    n_feat = int(why.split()[1].rstrip(","))
                    if n_feat < self.cfg.min_feature_inliers:
                        continue
                    fit = reg.evaluate_registration(keys[k].pcd, keys[r].pcd, self.cfg.icp_dist_m, T).fitness
                    C = (P[r] @ T) @ np.linalg.inv(P[k])      # k를 처음 장면 r에 맞추는 월드 보정
                    if best_r is None or n_feat * fit > best_r[3] * best_r[4]:
                        best_r = (r, k, C, n_feat, fit)
                if best_r:
                    found.append(best_r)
                    self.log(f"    처음 {ids.index(r) + 1}번째 장면(프레임 {keys[r].idx}) ↔ 가장 비슷한 끝 장면 "
                             f"프레임 {keys[best_r[1]].idx} (특징점 {best_r[3]}개, 겹침 {best_r[4]:.0%})")
            if not found:
                self.log(f"  패스 {pid + 1}: 처음 장면을 다시 본 프레임을 찾지 못함 (특징점 부족) → 마지막 연결 생략")
                continue
            # 2) 합의: 여러 쌍이 같은 보정을 가리키면 그게 맞다. 그중 가장 비슷한(특징점×겹침 최대) 쌍을 이음매로
            ok = lambda A, B: (lambda m, d: m < 0.015 and d < 3)(*_motion(np.linalg.inv(A) @ B))
            support = [sum(ok(f1[2], f2[2]) for f2 in found) for f1 in found]
            bi = max(range(len(found)), key=lambda t: (support[t], found[t][3] * found[t][4]))
            i0, k, C0 = found[bi][0], found[bi][1], found[bi][2]
            self.log(f"  패스 {pid + 1}: 처음 {len(found)}장과 순서대로 비교 → 가장 비슷한 쌍: 프레임 {keys[i0].idx} ↔ "
                     f"{keys[k].idx} ({len(found)}쌍 중 {support[bi]}쌍이 같은 보정에 동의)")
            # 돈 정도 = 연속 프레임 회전량을 쌓은 값 (항상 증가 → 각도 추정 오차에 영향 안 받음)
            path = {ids[0]: 0.0}
            for a, b in zip(ids[:-1], ids[1:]):
                path[b] = path[a] + _motion(np.linalg.inv(P[b]) @ P[a])[1]
            # 4) 보정 → 다시 재기 → 반복
            mid = min(late[0], k)          # 절반 지점 (뒤쪽 절반의 시작)
            near_k = [q for q in ids if abs(ids.index(q) - ids.index(k)) <= 2]
            C = C0
            for it in range(5):
                ref = self._world_cloud(keys, P, ref_ids)
                center = np.asarray(ref.points).mean(axis=0)
                if it > 0:
                    src = self._world_cloud(keys, P, near_k)
                    C, fit, rmse = self._icp_multistart(src, ref, center)
                    if fit < 0.3:
                        self.log(f"    {it + 1}회: 겹침 {fit:.0%}로 부족 → 중단")
                        break
                m, deg = _motion(C)
                self.log(f"    {it + 1}회: 남은 어긋남 {m * 1000:.1f}mm, {deg:.2f}도")
                if m < 0.001 and deg < 0.1:
                    break
                R = C[:3, :3]
                t_piv = C[:3, 3] - center + R @ center
                # 절반 지점까지는 그대로 두고, 절반 지점부터 끝까지 돈 정도에 비례해 0 → 1로 보정
                span = max(path[k] - path[mid], 1e-6)
                for q in ids:
                    frac = float(np.clip((path[q] - path[mid]) / span, 0.0, 1.0))
                    if frac <= 0:
                        continue
                    Rq = _interp_se3(np.vstack([np.hstack([R, np.zeros((3, 1))]), [0, 0, 0, 1]]), frac)[:3, :3]
                    D = np.eye(4)
                    D[:3, :3] = Rq
                    D[:3, 3] = center - Rq @ center + frac * t_piv
                    P[q] = D @ P[q]
            # 5) 처음 장면과 겹치는 뒤쪽 프레임을 하나씩 처음 장면에 맞춘다.
            #    누적 오차는 회전에 고르게 쌓이지 않으므로(사람도 조금씩 움직임) 비율로 나눈 보정만으로는
            #    이음매 프레임 하나만 맞고 그 앞뒤는 어긋난다 → 프레임마다 실제로 재서 각자 보정.
            P = self._align_overlap_frames(keys, P, ids, ref_ids, late, pid)
        return P

    def _align_overlap_frames(self, keys, P, ids, ref_ids, late, pid, rounds=4):
        from scipy.spatial.transform import Rotation
        ref = self._world_cloud(keys, P, ref_ids)
        center = np.asarray(ref.points).mean(axis=0)
        est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=0.005))
        for rnd in range(rounds):
            meas = {}       # 프레임 → (회전벡터, 중심 기준 이동)
            for q in late:
                src = o3d.geometry.PointCloud(keys[q].pcd).transform(P[q])
                if reg.evaluate_registration(src, ref, 0.012, np.eye(4)).fitness < 0.3:
                    continue
                src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
                T = np.eye(4)
                for dist in (0.03, 0.012, 0.005):
                    r = reg.registration_icp(src, ref, dist, T, est, reg.ICPConvergenceCriteria(max_iteration=30))
                    T = r.transformation
                # 맞춘 뒤 12mm 이내 겹침으로 판단 (5mm 기준이면 크게 어긋났던 프레임이 오히려 빠짐)
                if reg.evaluate_registration(src, ref, 0.012, T).fitness < 0.3:
                    continue
                rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
                tp = T[:3, 3] - center + T[:3, :3] @ center
                meas[q] = np.concatenate([rv, tp])
            if len(meas) < 3:
                if rnd == 0:
                    self.log(f"    겹치는 끝 프레임이 적어 프레임별 맞추기 생략 ({len(meas)}개)")
                return P
            mq = sorted(meas)
            vals = np.array([meas[q] for q in mq])
            # 처음엔 튀는 값을 앞뒤 프레임과의 중앙값으로 다듬고(큰 어긋남부터 안전하게),
            # 거의 맞은 뒤(마지막 회차)에는 프레임마다 자기 측정값을 그대로 적용
            if rnd < rounds - 1:
                sm = np.array([np.median(vals[max(0, t - 2):t + 3], axis=0) for t in range(len(mq))])
            else:
                sm = vals
            pos = {q: t for t, q in enumerate(ids)}
            xs = np.array([pos[q] for q in mq], float)
            x0 = float(pos[late[0]])          # 절반 지점: 보정 0
            # 겹치는 구간은 측정값, 그 앞(절반 지점~첫 겹침)은 0에서 첫 측정값까지 부드럽게, 뒤는 마지막 값 유지
            grid_x = np.concatenate([[x0], xs])
            grid_v = np.vstack([np.zeros(6), sm])
            mags = []
            for q in late:
                x = float(pos[q])
                v = np.array([np.interp(x, grid_x, grid_v[:, d]) for d in range(6)])
                Rq = Rotation.from_rotvec(v[:3]).as_matrix()
                D = np.eye(4)
                D[:3, :3] = Rq
                D[:3, 3] = center - Rq @ center + v[3:]
                P[q] = D @ P[q]
                mags.append(np.linalg.norm(v[3:]))
            off = np.linalg.norm(vals[:, 3:], axis=1) * 1000
            self.log(f"    프레임별 맞추기 {rnd + 1}회: 겹치는 끝 프레임 {len(mq)}개, "
                     f"어긋남 중앙값 {np.median(off):.1f}mm / 최대 {off.max():.1f}mm → 각각 보정")
            if np.median(off) < 1.0 and off.max() < 3.0:
                break
        return P

    @staticmethod
    def _small_perturbations(base, key):
        """작은 회전(±3·6도, 화면 세로축=회전의자 축 근처)과 작은 이동(±5mm) 후보."""
        out = [base]
        center = np.asarray(key.pcd.points).mean(axis=0) if len(key.pcd.points) else np.zeros(3)
        for deg in (-6, -3, 3, 6):
            for axis in ((0, 1, 0), (1, 0, 0)):
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array(axis, float) * np.deg2rad(deg))
                D = np.eye(4)
                D[:3, :3] = R
                D[:3, 3] = center - R @ center
                out.append(base @ D)
        for dx, dz in ((0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005)):
            D = np.eye(4)
            D[:3, 3] = (dx, 0, dz)
            out.append(base @ D)
        return out

    # ---------------- 그래프 도우미

    @staticmethod
    def _chain_poses(n, edges, alive=None):
        """정합들을 따라가며(BFS) 대략적 자세를 누적. 연결이 안 된 프레임은 None."""
        alive = set(range(n)) if alive is None else alive
        adj = {k: [] for k in alive}
        for (i, j), (T, _, _) in edges.items():
            if i in alive and j in alive:
                adj[i].append((j, np.linalg.inv(T)))   # P_j = P_i T^-1
                adj[j].append((i, T))                  # P_i = P_j T
        poses = [None] * n
        # 가까운 정합을 먼저 따라가도록 이웃을 프레임 간격 순으로
        for k in adj:
            adj[k].sort(key=lambda e: abs(e[0] - k))
        start = min(alive)
        poses[start] = np.eye(4)
        queue = [start]
        while queue:
            x = queue.pop(0)
            for y, M in adj[x]:
                if poses[y] is None:
                    poses[y] = poses[x] @ M
                    queue.append(y)
        return poses

    def _optimize(self, n, edges, alive, init_poses):
        ids = sorted(alive)
        nid = {k: t for t, k in enumerate(ids)}
        chain = self._chain_poses(n, edges, alive)
        pg = reg.PoseGraph()
        last = np.eye(4)
        for k in ids:
            if init_poses is not None and init_poses[k] is not None:
                p = init_poses[k]          # 이음매 보간으로 바로잡은 초기 자세
            else:
                p = chain[k] if chain[k] is not None else last
            last = p
            pg.nodes.append(reg.PoseGraphNode(p))
        for (i, j), (T, info, loop) in edges.items():
            if i in alive and j in alive:
                # 바로 옆(1칸)과 이음매 연결은 확실, 나머지는 불확실 → 최적화가 이상한 정합을 스스로 잘라냄
                if loop == "seam":
                    uncertain = False
                else:
                    uncertain = bool(loop) or j - i > 1
                pg.edges.append(reg.PoseGraphEdge(nid[i], nid[j], T, info, uncertain=uncertain))
        opt = reg.GlobalOptimizationOption(max_correspondence_distance=self.cfg.icp_dist_m,
                                           edge_prune_threshold=0.25, reference_node=0)
        reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                                reg.GlobalOptimizationConvergenceCriteria(), opt)
        P = [None] * n
        for k in ids:
            P[k] = np.asarray(pg.nodes[nid[k]].pose)
        return P

    def _bad_frames(self, edges, alive, P):
        """최적화된 자세와 정합 결과가 크게 어긋나는 쌍이 많은 프레임 = 잘못된 프레임."""
        c = self.cfg
        votes, total = {}, {}
        for (i, j), (T, _, _) in edges.items():
            if i not in alive or j not in alive:
                continue
            m, deg = _motion(np.linalg.inv(np.linalg.inv(P[j]) @ P[i]) @ T)
            bad = m > c.drop_trans_m or deg > c.drop_rot_deg
            for k in (i, j):
                total[k] = total.get(k, 0) + 1
                votes[k] = votes.get(k, 0) + bad
        return {k for k in alive if total.get(k, 0) == 0 or votes.get(k, 0) / total[k] > 0.5}

    @staticmethod
    def _connected_without(edges, alive, key):
        """key 연결을 빼도 두 끝이 여전히 이어져 있는가."""
        a, b = key
        adj = {}
        for (i, j) in edges:
            if (i, j) != key and i in alive and j in alive:
                adj.setdefault(i, []).append(j)
                adj.setdefault(j, []).append(i)
        seen, stack = {a}, [a]
        while stack:
            x = stack.pop()
            if x == b:
                return True
            for y in adj.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return False

    @staticmethod
    def _largest_component(edges, alive):
        adj = {k: set() for k in alive}
        for (i, j) in edges:
            if i in alive and j in alive:
                adj[i].add(j)
                adj[j].add(i)
        best, seen = set(), set()
        for s in alive:
            if s in seen:
                continue
            comp, stack = set(), [s]
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                stack.extend(adj[x] - comp)
            seen |= comp
            if len(comp) > len(best):
                best = comp
        return best

    def _frame_disagreement(self, keys, ids, P, v, f):
        """프레임별로 점들이 융합된 표면에서 frame_reject_far_m보다 멀리 떨어진 비율."""
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
        out = {}
        for i in ids:
            pts = np.asarray(keys[i].pcd.points)
            if len(pts) > 3000:
                pts = pts[np.random.default_rng(i).choice(len(pts), 3000, replace=False)]
            pw = (pts @ P[i][:3, :3].T + P[i][:3, 3]).astype(np.float32)
            d = sc.compute_distance(o3d.core.Tensor(pw)).numpy()
            out[i] = float(np.mean(d > self.cfg.frame_reject_far_m))
        return out

    def _refine_to_model(self, keys, ids, P, v, f):
        """각 프레임을 융합된 모델 표면에 point-to-plane ICP로 다시 맞춘다. 반환: 평균 이동량(m)."""
        c = self.cfg
        model = mt_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                                    o3d.utility.Vector3iVector(f.astype(np.int32)))
        mt_mesh.compute_vertex_normals()
        target = o3d.geometry.PointCloud(model.vertices)
        target.normals = model.vertex_normals
        target = target.voxel_down_sample(c.voxel_size * 0.75)
        est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=c.model_refine_dist_m))
        moves = []
        for i in ids:
            icp = reg.registration_icp(keys[i].pcd, target, c.model_refine_dist_m, P[i], est,
                                       reg.ICPConvergenceCriteria(max_iteration=20))
            if icp.fitness < 0.5:
                continue
            m, deg = _motion(np.linalg.inv(P[i]) @ icp.transformation)
            if m < 0.01 and deg < 3:       # 크게 튀는 보정은 믿지 않음
                P[i] = icp.transformation
                moves.append(m)
        return float(np.mean(moves)) if moves else 0.0

    def _integrate(self, keys, poses):
        c = self.cfg
        dev = o3c.Device("CPU:0")
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=c.voxel_size, block_resolution=16, block_count=50000, device=dev)
        intr = o3c.Tensor(self.K, o3c.float64)
        for k, P in zip(keys, poses):
            depth = o3d.t.geometry.Image(k.depth).to(dev)
            color = o3d.t.geometry.Image(k.color).to(dev)
            extr = o3c.Tensor(np.linalg.inv(P), o3c.float64)
            blocks = vbg.compute_unique_block_coordinates(depth, intr, extr, self.scale, c.depth_max)
            vbg.integrate(blocks, depth, color, intr, extr, self.scale, c.depth_max)
        m = vbg.extract_triangle_mesh(weight_threshold=c.weight_threshold).to_legacy()
        v = np.asarray(m.vertices).copy()
        f = np.asarray(m.triangles).astype(np.int64)
        col = np.asarray(m.vertex_colors).copy() if m.has_vertex_colors() else None
        if col is not None and col.max() > 1.5:
            col = col / 255.0
        return v, f, col
