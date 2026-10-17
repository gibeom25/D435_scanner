"""끌어서 모양 다듬기 (소프트 선택).

선택한 꼭짓점은 그대로(가중치 1) 움직이고, 선택하지 않은 주변은 표면을 따라 잰 거리에 따라
부드럽게 약해지며 따라온다(가중치 1 → 0). 표면을 따라 재므로 공간상 가깝지만 떨어진 부분
(귀와 머리 사이 등)은 같이 끌려오지 않는다.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


class Sculpt:
    MAX_HISTORY = 20

    def __init__(self, v, f, c):
        self.base = np.asarray(v, np.float64).copy()   # 현재 확정된 모양
        self.f = np.asarray(f, np.int64)
        self.c = np.asarray(c, np.float64)
        self.n = len(self.base)
        self.sel = np.zeros(self.n, bool)
        self.w = np.zeros(self.n)
        self.radius = 0.04
        self.history = []
        e = np.concatenate([self.f[:, [0, 1]], self.f[:, [1, 2]], self.f[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        self.edges = e
        self._rebuild()

    def _rebuild(self):
        """모양이 바뀐 뒤: 거리 그래프와 붓 검색용 트리를 다시 만든다."""
        e = self.edges
        length = np.linalg.norm(self.base[e[:, 0]] - self.base[e[:, 1]], axis=1) + 1e-9
        self.graph = sparse.csr_matrix((length, (e[:, 0], e[:, 1])), shape=(self.n, self.n))
        self.tree = cKDTree(self.base)
        self.normals = self._vertex_normals()

    def _vertex_normals(self):
        v, f = self.base, self.f
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)

    # ---------------- 선택

    def select(self, point, brush, add=True):
        idx = self.tree.query_ball_point(point, brush)
        if not idx:
            return 0
        self.sel[idx] = add
        self.update_weights()
        return len(idx)

    def clear(self):
        self.sel[:] = False
        self.w[:] = 0

    def set_radius(self, r):
        self.radius = max(r, 0.0)
        self.update_weights()

    def update_weights(self):
        """선택=1, 주변은 표면을 따라 잰 거리 d에 따라 (1-(d/R)^2)^2로 부드럽게 0까지."""
        self.w[:] = 0
        src = np.flatnonzero(self.sel)
        if len(src) == 0:
            return
        if self.radius > 0:
            d = dijkstra(self.graph, directed=False, indices=src, min_only=True, limit=self.radius)
            t = np.clip(d / self.radius, 0, 1)
            near = np.isfinite(d)
            self.w[near] = (1 - t[near] ** 2) ** 2
        self.w[src] = 1.0

    def selection_normal(self):
        if not self.sel.any():
            return np.zeros(3)
        n = self.normals[self.sel].mean(axis=0)
        return n / (np.linalg.norm(n) + 1e-12)

    def affected(self):
        return self.w > 0

    # ---------------- 변형

    def preview(self, move):
        """move(3,) 만큼 끌었을 때의 꼭짓점 위치 (확정하지 않음)."""
        return self.base + self.w[:, None] * np.asarray(move)[None, :]

    def commit(self, move):
        if not np.any(move) or not self.w.any():
            return
        self.history.append(self.base.copy())
        del self.history[:-self.MAX_HISTORY]
        self.base = self.preview(move)
        self._rebuild()

    # ---------------- 가우시안 스무딩

    def _smooth_operator(self):
        """이웃 평균 연산자(표면 연결 기준)와 열린 가장자리 꼭짓점. 한 번 만들어 재사용."""
        if getattr(self, "_W", None) is not None:
            return self._W, self._boundary
        e = self.edges
        A = sparse.csr_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                              shape=(self.n, self.n))
        A.data[:] = 1.0
        deg = np.asarray(A.sum(axis=1)).ravel()
        self._W = sparse.diags(1.0 / np.maximum(deg, 1)) @ A
        # 열린 가장자리: 한 삼각형에만 속한 변의 꼭짓점
        f = self.f
        fe = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, cnt = np.unique(fe, axis=0, return_counts=True)
        self._boundary = np.zeros(self.n, bool)
        self._boundary[uniq[cnt == 1].ravel()] = True
        return self._W, self._boundary

    def gaussian_smooth(self, sigma_m, selection_only=False, keep_volume=True):
        """표면을 따라 퍼지는 가우시안 스무딩 (이웃 평균을 반복 = 열 확산 ≈ 가우시안).
        sigma_m: 퍼지는 폭(m). 평균 변 길이로 반복 횟수를 정한다.
        selection_only: 선택 부분만 (주변은 영향 반경 가중치만큼 점점 약하게).
        keep_volume: Taubin 방식(+λ, -μ 번갈아)으로 쪼그라듦 상쇄.
        반환: 바뀐 꼭짓점 수. 확정(되돌리기 가능)까지 한다."""
        W, boundary = self._smooth_operator()
        e = self.edges
        edge_len = float(np.mean(np.linalg.norm(self.base[e[:, 0]] - self.base[e[:, 1]], axis=1)))
        lam, mu = 0.5, -0.53
        # 한 번의 이웃 평균(λ)은 분산을 약 λ·ℓ² 만큼 퍼뜨린다 → σ² 만큼 퍼지려면 (σ/ℓ)²/λ 회
        iters = int(np.clip(np.ceil((sigma_m / max(edge_len, 1e-6)) ** 2 / lam), 1, 400))
        if selection_only:
            if not self.sel.any():
                return 0
            wgt = self.w.copy()
        else:
            wgt = np.ones(self.n)
        wgt[boundary] = 0.0                 # 열린 가장자리는 고정
        x = self.base.copy()
        for k in range(iters):
            x = x + lam * wgt[:, None] * (W @ x - x)
            if keep_volume:
                x = x + mu * wgt[:, None] * (W @ x - x)
        moved = int(np.sum(np.linalg.norm(x - self.base, axis=1) > 1e-7))
        if moved:
            self.history.append(self.base.copy())
            del self.history[:-self.MAX_HISTORY]
            self.base = x
            self._rebuild()
        self.last_iters = iters
        return moved

    def undo(self):
        if not self.history:
            return False
        self.base = self.history.pop()
        self._rebuild()
        return True

    def display_colors(self):
        """선택=주황, 따라오는 영역=가중치만큼 노랗게."""
        col = self.c.copy()
        a = self.w[:, None] * 0.6
        col = col * (1 - a) + np.array([1.0, 0.8, 0.2]) * a
        col[self.sel] = [1.0, 0.45, 0.0]
        return col
