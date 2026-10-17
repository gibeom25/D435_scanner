"""RealSense D435 핸드헬드 스캔: 프레임-모델 추적 + TSDF 융합 (Open3D t.pipelines.slam).

미리보기 창 조작
  SPACE : 스캔 시작 / 일시정지
  R     : 처음부터 다시
  ENTER : 스캔 끝내고 메시 만들기
  ESC   : 저장하지 않고 종료
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pyrealsense2 as rs

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # 자세 갱신 경고 스팸 방지

WIN = "RealSense Scan  [SPACE start/pause | R reset | ENTER finish | ESC quit]"


@dataclass
class ScanConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    voxel_size: float = 0.004       # 4mm. 작은 물체면 0.002~0.003, 큰 공간이면 0.01
    depth_min: float = 0.15
    depth_max: float = 1.0          # 이보다 먼 곳은 무시 (배경 잡음 차단)
    trunc_multiplier: float = 8.0
    block_count: int = 40000
    # 추적 성공 판정. fitness는 '깊이가 있는 픽셀 중 모델과 대응된 비율'로 환산해서 본다.
    # (Open3D fitness는 전체 픽셀 대비라 물체가 화면 일부만 차지하면 낮게 나옴)
    # 새로 보는 영역이 많아도 겹치는 부분만 잘 맞으면 성공으로 보고 융합해야 모델이 자란다.
    min_overlap: float = 0.25
    # 피라미드(거친→세밀) 단계별 ICP 반복 횟수. Open3D 기본값(6,3,1)은 손으로 들고 움직일 때
    # 프레임당 이동을 절반 정도만 따라가 오차가 쌓이고 결국 추적을 놓친다.
    track_iters: tuple = (6, 3, 1)
    # 깊이만 쓰는 정합(PointToPlane)은 책상 같은 넓은 평면이 많으면 옆으로 미끄러지는 움직임을
    # 거의 못 잡는다. 컬러도 함께 쓰는 Hybrid가 훨씬 정확.
    track_method: str = "hybrid"          # "hybrid" | "point_to_plane"
    max_rmse_m: float = 0.015
    min_valid_ratio: float = 0.05   # 사용 거리 안의 깊이 픽셀이 이보다 적으면 처리하지 않음
    max_step_m: float = 0.08        # 한 프레임에 이보다 많이 움직이면 실패로 간주
    max_step_deg: float = 15.0
    # 놓쳤을 때: 연속 프레임끼리 비교(RGB-D 오도메트리)로 위치를 계속 추정하고,
    # 그 위치에서 모델을 다시 렌더링해 재정합한다. 이 거리/각도 이상 흘러가면 마지막 성공 위치로 되돌림.
    max_drift_m: float = 0.5
    max_drift_deg: float = 90.0
    # 사람 스캔 모드 (카메라 고정 + 회전의자): 대상 거리 ±subject_half_depth만 남기고 배경 제거
    subject_mode: bool = False
    # 깊이 프리셋. high_accuracy는 불확실한 픽셀을 버려 노이즈가 약 40% 적다(실측 3.7→2.2mm).
    # 검은 머리카락 등에서 구멍이 많이 생기면 "default"나 "high_density"로.
    preset: str = "high_accuracy"
    subject_half_depth: float = 0.3


class RealSense:
    def __init__(self, cfg: ScanConfig):
        if len(rs.context().query_devices()) == 0:
            raise RuntimeError(
                "RealSense 카메라를 찾지 못했습니다.\n"
                "  - USB 3 포트(파란색)에 직접 꽂았는지, 케이블이 헐겁지 않은지 확인하세요.\n"
                "  - 다른 프로그램(RealSense Viewer, 카메라 앱 등)이 사용 중이면 닫으세요.")
        d0 = rs.context().query_devices()[0]
        usb = d0.get_info(rs.camera_info.usb_type_descriptor) \
            if d0.supports(rs.camera_info.usb_type_descriptor) else "?"
        fps = 15 if usb.startswith("2") else cfg.fps   # USB 2는 대역폭이 부족해 프레임이 빠짐
        self.usb, self.fps = usb, fps
        self.pipe = rs.pipeline()
        c = rs.config()
        c.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, fps)
        c.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.rgb8, fps)
        profile = self.pipe.start(c)

        dev = profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        presets = {"default": 1, "high_accuracy": 3, "high_density": 4}
        if depth_sensor.supports(rs.option.visual_preset) and cfg.preset in presets:
            depth_sensor.set_option(rs.option.visual_preset, presets[cfg.preset])
        if depth_sensor.supports(rs.option.laser_power):
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, rng.max)

        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.holes_fill, 0)   # 구멍을 지어내 채우지 않음 (경계 가짜 깊이 방지)

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])

        # 자동 노출이 안정될 때까지 몇 프레임 버림
        for _ in range(15):
            self.pipe.wait_for_frames()

    def read(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        d, c = frames.get_depth_frame(), frames.get_color_frame()
        if not d or not c:
            return None, None
        d = self.spatial.process(d)
        depth = np.asanyarray(d.get_data()).copy()
        color = np.asanyarray(c.get_data()).copy()
        return depth, color

    def close(self):
        self.pipe.stop()


class Fusion:
    """Open3D 밀집 SLAM 래퍼 (CPU)."""

    def __init__(self, cfg: ScanConfig, K, depth_scale_factor):
        self.cfg = cfg
        self.device = o3c.Device("CUDA:0" if o3c.cuda.is_available() else "CPU:0")
        self.K = o3c.Tensor(K, o3c.float64)
        Kh = np.array(K, float).copy()
        Kh[:2] *= 0.5
        self.K_half = o3c.Tensor(Kh, o3c.float64)
        self.scale = depth_scale_factor   # raw → m 를 나누는 값 (보통 1000)
        self.reset()

    def reset(self):
        c = self.cfg
        self.model = o3d.t.pipelines.slam.Model(
            c.voxel_size, 16, c.block_count, o3c.Tensor(np.eye(4)), self.device)
        self.input_frame = o3d.t.pipelines.slam.Frame(c.height, c.width, self.K, self.device)
        self.raycast_frame = o3d.t.pipelines.slam.Frame(c.height, c.width, self.K, self.device)
        self.T = np.eye(4)          # 직전 프레임의 (추정) 카메라 자세
        self.T_good = np.eye(4)     # 마지막으로 모델에 정합 성공한 자세
        self.raycast_T = np.eye(4)  # raycast_frame을 렌더링한 자세
        self.prev_rgbd = None       # 직전 프레임 (연속 프레임 오도메트리용, 절반 해상도)
        self.vel = np.eye(4)        # 직전 프레임 간 움직임 (등속 예측)
        self.blind_frames = 0       # 깊이가 끊긴 프레임 수
        self.lost_streak = 0
        self.index = 0
        self.integrated = 0
        self.relocalized = 0

    # ---------------------------------------------------------------- 내부

    @staticmethod
    def _motion(delta):
        m = np.linalg.norm(delta[:3, 3])
        deg = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
        return m, deg

    def _raycast_at(self, T):
        """모델을 자세 T에서 렌더링해 raycast_frame에 담는다."""
        c = self.cfg
        self.model.update_frame_pose(self.index, o3c.Tensor(T))
        self.model.synthesize_model_frame(self.raycast_frame, self.scale, c.depth_min,
                                          c.depth_max, c.trunc_multiplier, True)
        self.raycast_T = T.copy()

    def _track_model(self, valid, ref=None, return_score=False):
        """현재 입력을 raycast_frame(raycast_T 시점의 모델)에 정합. 반환: (자세|None, 정보)
        ref: 직전 프레임 자세. 결과가 여기서 너무 멀면(순간이동) 실패로 본다."""
        c = self.cfg
        try:
            res = self.model.track_frame_to_model(
                self.input_frame, self.raycast_frame, self.scale, c.depth_max,
                method=(o3d.t.pipelines.odometry.Method.Hybrid if c.track_method == "hybrid"
                        else o3d.t.pipelines.odometry.Method.PointToPlane),
                criteria=[o3d.t.pipelines.odometry.OdometryConvergenceCriteria(n) for n in c.track_iters])
        except RuntimeError:
            return (None, "no overlap", 0.0) if return_score else (None, "no overlap")
        delta = res.transformation.cpu().numpy()
        T = self.raycast_T @ delta
        # 움직임 제한은 '예측 위치'가 아니라 '직전 프레임 위치' 기준 (예측이 틀려도 모델 정합이 바로잡은 건 인정)
        m, deg = self._motion(np.linalg.inv(ref) @ T) if ref is not None else (0.0, 0.0)
        k = 1 + min(self.lost_streak, 10)   # 놓친 동안엔 그만큼 더 움직였을 수 있음
        overlap = res.fitness / max(valid, 1e-6)
        info = f"overlap={overlap:.2f} rmse={res.inlier_rmse * 1000:.1f}mm"
        bad = ("low overlap" if overlap < c.min_overlap else
               "high error" if res.inlier_rmse > c.max_rmse_m else
               f"jump {m * 1000:.0f}mm/{deg:.0f}deg" if m > k * c.max_step_m or deg > k * c.max_step_deg
               else None)
        if bad:
            info += f" [{bad}]"
            return (None, info, 0.0) if return_score else (None, info)
        # 점수: 겹침이 클수록, 오차가 작을수록 좋음
        return (T, info, overlap - 20 * res.inlier_rmse) if return_score else (T, info)

    def _half(self, color_img, depth_img):
        """프레임 간 오도메트리용 절반 해상도 RGBD (속도 4배)."""
        Interp = o3d.t.geometry.InterpType
        return o3d.t.geometry.RGBDImage(color_img.resize(0.5, Interp.Linear),
                                        depth_img.resize(0.5, Interp.Nearest))

    def _track_frame_to_frame(self, cur_rgbd, init=np.eye(4)):
        """직전 프레임 대비 움직임 (T_prev_cur). 실패하면 None."""
        if self.prev_rgbd is None:
            return None
        c = self.cfg
        try:
            res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                cur_rgbd, self.prev_rgbd, self.K_half, o3c.Tensor(init), self.scale, c.depth_max,
                method=o3d.t.pipelines.odometry.Method.Hybrid)
        except RuntimeError:
            return None
        delta = res.transformation.cpu().numpy()
        m, deg = self._motion(delta)
        if res.fitness < 0.05 or m > 2 * c.max_step_m or deg > 2 * c.max_step_deg:
            return None
        return delta

    def _reacquire(self, valid, depth):
        """깊이가 끊겼다가(가림 등) 돌아온 첫 프레임: 그 사이 움직임을 알 수 없으므로
        마지막 자세 주변을 물체 중심으로 공전시킨 후보 자세들에서 각각 정합해 가장 잘 맞는 것을 고른다."""
        c = self.cfg
        dm = depth[depth > 0].astype(np.float32) / self.scale
        dm = dm[(dm > c.depth_min) & (dm < c.depth_max)]
        pivot = np.eye(4)
        pivot[2, 3] = float(np.median(dm)) if len(dm) else 0.5   # 화면 중앙 깊이 = 물체 중심 근사
        best = (None, "no overlap", -1.0)
        bases = [self.T] if np.allclose(self.T, self.T_good) else [self.T, self.T_good]
        for base, yaw, pitch in [(B, y, p) for B in bases
                                 for y in (0, -8, 8, -16, 16, -24, 24) for p in (0, -8, 8)]:
            a, b = np.deg2rad(yaw), np.deg2rad(pitch)
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
            R = np.eye(4)
            R[:3, :3] = Ry @ Rx
            cand = base @ pivot @ R @ np.linalg.inv(pivot)
            self._raycast_at(cand)
            T_new, info, score = self._track_model(valid, return_score=True)
            if T_new is not None and score > best[2]:
                best = (T_new, info, score)
        return best[0], best[1]

    # ---------------------------------------------------------------- 공개

    def step(self, depth, color):
        """한 프레임 처리. 반환: (ok, 메시지)"""
        c = self.cfg
        depth_img = o3d.t.geometry.Image(depth).to(self.device)
        color_img = o3d.t.geometry.Image(color).to(self.device)
        self.input_frame.set_data_from_image("depth", depth_img)
        self.input_frame.set_data_from_image("color", color_img)

        # 사용 거리 안의 깊이가 거의 없으면(너무 멀거나 가까움) 추적/융합하지 않는다.
        # 특히 첫 프레임이 비어 있으면 모델이 비어 이후 추적이 전부 실패하므로 꼭 막아야 한다.
        dm = depth.astype(np.float32) / self.scale
        valid = float(((dm > c.depth_min) & (dm < c.depth_max)).mean())
        if valid < c.min_valid_ratio:
            self.prev_rgbd = None
            if self.integrated:
                # 가려진 동안에도 손은 계속 움직이므로 직전 속도로 위치를 외삽 (최대 1초)
                self.blind_frames += 1
                if self.blind_frames <= 15:
                    self.T = self.T @ self.vel
            return False, f"NO DEPTH in range ({valid:.0%}) - move to {c.depth_min:.2f}-{c.depth_max:.1f} m"
        cur_rgbd = self._half(color_img, depth_img)

        if self.integrated == 0:
            T_new, msg = np.eye(4), "OK first frame"
        else:
            # 1) 직전 프레임과 컬러+깊이로 비교해 이번 움직임을 예측 (직전 속도를 초기값으로).
            #    빠르게 움직여도 모델 정합이 엉뚱한 곳에 붙지 않게 하고,
            #    놓친 동안에도 위치를 계속 따라가 원래 자리로 돌아가지 않아도 다시 붙게 한다.
            delta = self._track_frame_to_frame(cur_rgbd, self.vel)
            est = self.T @ delta if delta is not None else self.T @ self.vel
            if self.lost_streak:
                # 놓친 상태로 너무 멀리 흘러갔으면 마지막 성공 위치로 되돌림
                m, deg = self._motion(np.linalg.inv(self.T_good) @ est)
                if m > c.max_drift_m or deg > c.max_drift_deg:
                    est = self.T_good.copy()
            # 2) 예측 자세에서 렌더링한 모델에 정합해 오차 보정 (드리프트 방지)
            if self.prev_rgbd is None:
                # 깊이가 끊겼다 돌아옴 → 움직임을 모르므로 후보 자세 여러 개로 재탐색
                T_new, info = self._reacquire(valid, depth)
            else:
                if not np.allclose(self.raycast_T, est):
                    self._raycast_at(est)
                T_new, info = self._track_model(valid, ref=self.T)
                pred = self.T @ self.vel
                if T_new is None and delta is not None and not np.allclose(pred, est):
                    # 프레임 간 예측이 틀렸을 수 있음 → 등속 예측 위치에서 한 번 더
                    self._raycast_at(pred)
                    T_new, info2 = self._track_model(valid, ref=self.T)
                    if T_new is not None:
                        est, info = pred, info2
            if T_new is None:
                self.T = est
                self.vel = delta if delta is not None else np.eye(4)
                self.prev_rgbd = cur_rgbd
                self.lost_streak += 1
                how = "following" if delta is not None else "hold"
                return False, f"LOST {info} ({how}) - keep moving slowly"
            msg = f"OK {info}"
            if self.lost_streak:
                self.relocalized += 1
                msg = f"RECOVERED {info}"

        if self.integrated and not self.blind_frames:
            self.vel = np.linalg.inv(self.T) @ T_new   # 다음 프레임 예측용 (등속 가정)
        self.blind_frames = 0
        self.T = self.T_good = T_new
        self.lost_streak = 0
        self.prev_rgbd = cur_rgbd
        self.model.update_frame_pose(self.index, o3c.Tensor(T_new))
        self.model.integrate(self.input_frame, self.scale, c.depth_max, c.trunc_multiplier)
        self.model.synthesize_model_frame(self.raycast_frame, self.scale, c.depth_min,
                                          c.depth_max, c.trunc_multiplier, True)
        self.raycast_T = T_new.copy()
        self.index += 1
        self.integrated += 1
        return True, msg

    def model_view(self):
        """현재까지 복원된 모델을 카메라 시점에서 렌더링한 이미지 (BGR uint8)."""
        if self.integrated == 0:
            return None
        img = self.raycast_frame.get_data_as_image("color").cpu().as_tensor().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def extract_mesh(self):
        """(vertices, faces, colors) numpy, 카메라 첫 프레임 좌표계, 단위 m."""
        m = self.model.extract_trianglemesh(weight_threshold=3.0).to_legacy()
        v = np.asarray(m.vertices).copy()
        f = np.asarray(m.triangles).astype(np.int64)
        col = np.asarray(m.vertex_colors).copy() if m.has_vertex_colors() else None
        if col is not None and col.max() > 1.5:
            col = col / 255.0
        return v, f, col


class SubjectMask:
    """사람 스캔 모드: 카메라는 고정, 사람이 회전의자에서 돈다.
    화면 가운데 있는 대상의 거리 ±half_depth만 남기고 배경을 지운다.
    (배경이 남아 있으면 추적이 가만히 있는 배경에 맞춰져 회전하는 사람이 잔상처럼 겹친다)"""

    def __init__(self, depth_scale, half_depth=0.3):
        self.scale = depth_scale
        self.half = half_depth
        self.center_m = None    # 잠긴 대상 거리 (None이면 매 프레임 다시 잼)

    def measure(self, depth):
        h, w = depth.shape
        roi = depth[int(h * 0.25):int(h * 0.75), int(w * 0.35):int(w * 0.65)]
        roi = roi[roi > 0] * self.scale
        return float(np.median(roi)) if len(roi) > 100 else None

    def lock(self, depth):
        self.center_m = self.measure(depth)
        return self.center_m

    def unlock(self):
        self.center_m = None

    def apply(self, depth):
        """배경을 0으로 지운 깊이 영상과 마스크를 반환."""
        center = self.center_m if self.center_m is not None else self.measure(depth)
        if center is None:
            return np.zeros_like(depth), np.zeros(depth.shape, bool)
        dm = depth * self.scale
        mask = (dm > center - self.half) & (dm < center + self.half)
        # 범위 안에 걸친 다른 물체(옆 벽 등) 제거: 화면 가운데와 가장 많이 겹치는 덩어리만
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if n > 2:
            h, w = depth.shape
            center_labels = labels[int(h * 0.25):int(h * 0.75), int(w * 0.35):int(w * 0.65)]
            counts = np.bincount(center_labels.ravel(), minlength=n)
            counts[0] = 0
            keep = counts.argmax() if counts.max() > 0 else stats[1:, cv2.CC_STAT_AREA].argmax() + 1
            mask = labels == keep
        # 실루엣 경계의 튀는 깊이(귀·머리카락 옆 줄무늬)를 줄이려 경계를 2픽셀 깎음
        mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        out = depth.copy()
        out[~mask] = 0
        return out, mask


def _overlay(img, lines, color):
    y = 24
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 24


@dataclass
class ScanResult:
    mesh: tuple             # 실시간 추적으로 만든 (vertices, faces, colors)
    recording: "Recording"


class Recording:
    """정밀 재구성용 녹화. 깊이·컬러 모두 무손실 PNG로 압축해 보관 (예전 JPEG 녹화도 그대로 읽힘)."""

    def __init__(self, K, depth_scale_factor, subject_mode):
        self.K = np.asarray(K, float)
        self.scale = depth_scale_factor
        self.subject_mode = subject_mode
        self.depth_min = self.depth_max = None   # 스캔할 때 쓴 거리 범위 (m). 예전 녹화에는 없음
        self.depth, self.color, self.poses, self.passes = [], [], [], []
        self._pool = None

    @staticmethod
    def _encode(depth, color):
        d = cv2.imencode(".png", depth)[1]
        # 컬러도 무손실 PNG (예전엔 JPEG 92 → 질감·특징점에 약간 손해). 압축 1단계로 빠르게.
        c = cv2.imencode(".png", cv2.cvtColor(color, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])[1]
        return d, c

    def add(self, depth, color, pose, pass_id):
        """압축은 백그라운드 스레드에서 (스캔 속도에 영향 없게)."""
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=2)
        fut = self._pool.submit(self._encode, depth, color)
        self.depth.append(fut)
        self.color.append(None)
        self.poses.append(np.array(pose))
        self.passes.append(pass_id)

    def _resolve(self, i):
        d = self.depth[i]
        if hasattr(d, "result"):
            self.depth[i], self.color[i] = d.result()

    def __len__(self):
        return len(self.depth)

    def frame(self, i):
        self._resolve(i)
        d = cv2.imdecode(self.depth[i], cv2.IMREAD_UNCHANGED)
        c = cv2.cvtColor(cv2.imdecode(self.color[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return d, c

    def save(self, path):
        """한 파일(.npz)로 저장. 나중에 다시 정밀 재구성하거나 문제를 분석할 때 사용."""
        for i in range(len(self.depth)):
            self._resolve(i)
        arrs = {f"d{i:05d}": b for i, b in enumerate(self.depth)}
        arrs.update({f"c{i:05d}": b for i, b in enumerate(self.color)})
        with open(path, "wb") as fh:   # 한글 경로 대응: 파일 핸들로 저장
            if self.depth_max is not None:
                arrs["depth_range"] = np.array([self.depth_min, self.depth_max], float)
            np.savez(fh, K=self.K, scale=self.scale, subject=self.subject_mode,
                     poses=np.array(self.poses), passes=np.array(self.passes), **arrs)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            z = np.load(fh)
            rec = cls(z["K"], float(z["scale"]), bool(z["subject"]))
            n = len(z["poses"])
            rec.depth = [z[f"d{i:05d}"] for i in range(n)]
            rec.color = [z[f"c{i:05d}"] for i in range(n)]
            rec.poses = list(z["poses"])
            rec.passes = list(z["passes"])
            if "depth_range" in z.files:
                rec.depth_min, rec.depth_max = (float(x) for x in z["depth_range"])
        return rec

    def refine_depth_range(self):
        """정밀 재구성에 쓸 거리 범위 (m). 스캔 때 정한 범위, 없으면(예전 녹화) 예전 기본값."""
        if self.depth_max is not None:
            return self.depth_min, self.depth_max
        return 0.0, (2.0 if self.subject_mode else 1.0)


def run_scan(cfg: ScanConfig):
    """스캔 창을 띄우고, 끝나면 ScanResult를 반환. 취소하면 None."""
    cam = RealSense(cfg)
    print(f"카메라 연결됨 (USB {cam.usb}). 장치: {'GPU' if o3c.cuda.is_available() else 'CPU'}")
    if cam.usb.startswith("2"):
        print(f"경고: USB {cam.usb}로 연결되어 있어 {cam.fps}fps로 낮췄습니다. "
              "USB 3 포트(파란색)/USB 3 케이블을 쓰면 훨씬 안정적입니다.")

    fusion = Fusion(cfg, cam.K, 1.0 / cam.depth_scale)
    rec = Recording(cam.K, 1.0 / cam.depth_scale, cfg.subject_mode)
    pass_id = -1   # 일시정지 후 다시 시작할 때마다 새 패스 (카메라 위치를 바꿔 위쪽도 찍을 수 있게)
    subject = SubjectMask(cam.depth_scale, cfg.subject_half_depth) if cfg.subject_mode else None
    if subject:
        print("사람 스캔 모드: 카메라를 고정하고, 화면 가운데 앉아 SPACE를 누른 뒤 "
              "회전의자로 30~40초에 한 바퀴 천천히 도세요.")
    running, last_ok, msg = False, True, "Press SPACE to start"
    t_start = None
    processed = lost = 0
    t_last = time.time()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    try:
        while True:
            depth, color = cam.read()
            if depth is None:
                continue
            mask = None
            if subject:
                depth, mask = subject.apply(depth)
            if running:
                was_ok = last_ok
                last_ok, msg = fusion.step(depth, color)
                rec.add(depth, color, fusion.T, pass_id)
                processed += 1
                if not last_ok:
                    lost += 1
                    if was_ok:
                        print(f"  추적 놓침 (프레임 {fusion.integrated}): {msg}")
                elif not was_ok:
                    print(f"  추적 복구 (프레임 {fusion.integrated})")
                if time.time() - t_last > 3:
                    fps = processed / (time.time() - t_last)
                    print(f"  융합 {fusion.integrated}프레임, {fps:.1f} fps, 최근 놓침 {lost}/{processed}")
                    processed = lost = 0
                    t_last = time.time()

            live = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            # 사용 범위 밖(너무 가깝거나 먼) 영역은 어둡게
            if mask is not None:
                out = ~mask       # 사람 모드: 스캔 대상(남는 부분)만 밝게
            else:
                dm = depth * cam.depth_scale
                out = (dm < cfg.depth_min) | (dm > cfg.depth_max)
            live[out] = (live[out] * 0.3).astype(np.uint8)

            state = "SCANNING" if running else "PAUSED"
            col = (0, 220, 0) if (last_ok or not running) else (0, 0, 255)
            if subject:
                center = subject.center_m if subject.center_m is not None else subject.measure(depth)
                rng = f"subject {center:.2f} m +-{cfg.subject_half_depth:.1f}" if center else "no subject in center"
            else:
                rng = f"range {cfg.depth_min:.2f}-{cfg.depth_max:.2f} m   [ ] far  , . near"
            elapsed = f"  {time.time() - t_start:.0f}s" if running and t_start else ""
            _overlay(live, [f"{state}  frames={fusion.integrated}{elapsed}", msg, rng], col)

            mv = fusion.model_view()
            if mv is None:
                mv = np.zeros_like(live)
                _overlay(mv, ["model view"], (200, 200, 200))
            if not last_ok and running:
                cv2.rectangle(live, (0, 0), (live.shape[1] - 1, live.shape[0] - 1), (0, 0, 255), 8)
            cv2.imshow(WIN, np.hstack([live, mv]))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("ESC: 취소")
                return None
            if key == ord(" "):
                running = not running
                if running:
                    pass_id += 1
                    if pass_id > 0:
                        print(f"패스 {pass_id + 1} 시작 (카메라 위치를 바꿨다면 정밀 재구성에서 이어 붙입니다)")
                if running and t_start is None:
                    t_start = time.time()
                if running and subject and subject.center_m is None:
                    # 원본 깊이로 대상 거리를 잠근다 (회전해도 거리는 그대로)
                    raw, _ = cam.read()
                    c_m = subject.lock(raw) if raw is not None else None
                    print(f"대상 거리 고정: {c_m:.2f} m" if c_m else "대상을 찾지 못했습니다")
                print("스캔 시작" if running else "일시정지")
                t_last, processed, lost = time.time(), 0, 0
                msg = "scanning..." if running else "paused"
            elif key in (ord("r"), ord("R")):
                fusion.reset()
                rec = Recording(cam.K, 1.0 / cam.depth_scale, cfg.subject_mode)
                pass_id = -1
                t_start = None
                if subject:
                    subject.unlock()
                print("처음부터 다시")
                running, last_ok, msg = False, True, "reset - press SPACE"
            elif not subject and key in (ord("["), ord("]"), ord(","), ord(".")):
                # 거리 범위 조절 (5cm씩). 범위 밖은 화면에서 어둡게 보이고 스캔에서 빠진다.
                step = 0.05
                if key == ord("]"):
                    cfg.depth_max = min(3.0, cfg.depth_max + step)
                elif key == ord("["):
                    cfg.depth_max = max(cfg.depth_min + 0.1, cfg.depth_max - step)
                elif key == ord("."):
                    cfg.depth_min = min(cfg.depth_max - 0.1, cfg.depth_min + step)
                else:
                    cfg.depth_min = max(0.1, cfg.depth_min - step)
                cfg.depth_min, cfg.depth_max = round(cfg.depth_min, 2), round(cfg.depth_max, 2)
                print(f"거리 범위: {cfg.depth_min:.2f} ~ {cfg.depth_max:.2f} m")
            elif key in (13, 10):
                if fusion.integrated < 5:
                    msg = "need more frames"
                    continue
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cam.close()
        cv2.destroyAllWindows()

    rec.depth_min, rec.depth_max = cfg.depth_min, cfg.depth_max
    print(f"메시 추출 중... (융합 프레임 {fusion.integrated}개)")
    return ScanResult(fusion.extract_mesh(), rec)
