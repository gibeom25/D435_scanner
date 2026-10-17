"""RealSense 3D 스캐너.

  python scanner.py                      스캔 → 편집기
  python scanner.py --open 경로.ply       예전 스캔을 편집기로 열기
  python scanner.py --voxel 0.003 --max-depth 0.8   작은 물체용 세밀 스캔
"""
import argparse
import os
import sys
from datetime import datetime

# exe로 묶였을 때는 __file__이 임시 폴더라서, 실행 파일이 있는 폴더를 기준으로 한다
HERE = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
# 런처가 저장 폴더·작업별 로그 파일을 환경 변수로 넘겨준다
SCANS = os.environ.get("RS_SCANS_DIR") or os.path.join(HERE, "scans")
LOG_PATH = os.environ.get("RS_LOG_PATH") or os.path.join(HERE, "logs", "last_run.log")


class _Tee:
    """콘솔 출력과 오류를 logs/last_run.log에도 남긴다 (실행할 때마다 새로 씀)."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, s):
        if self.stream is not None:      # 창 모드 exe에는 콘솔이 없음
            self.stream.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        if self.stream is not None:
            self.stream.flush()
        self.file.flush()


def _start_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    f = open(LOG_PATH, "w", encoding="utf-8")
    f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {' '.join(sys.argv)}\n")
    f.flush()
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)


def _refine_recordings(recs, voxel, out_dir, seam_ref=None):
    """녹화(여러 개면 각각 별도 패스로)를 정밀 재구성해 scene.ply로 저장 (기존 결과는 scene_prev.ply로 보관).
    voxel: 정밀 재구성 복셀 크기 m. seam_ref: 마지막 연결에서 끝과 순서대로 비교할 처음 장면 수.
    반환: (v, f, c) 또는 None."""
    import mesh_tools as mt
    from refine import Refiner, RefineConfig

    K, scale, subject = recs[0].K, recs[0].scale, any(r.subject_mode for r in recs)
    index, passes, poses = [], [], []
    offset = 0
    for ri, r in enumerate(recs):
        for i in range(len(r)):
            index.append((ri, i))
            passes.append(offset + int(r.passes[i]))
            poses.append(r.poses[i])
        offset += int(max(r.passes)) + 1 if len(r) else 0
    get = lambda k: recs[index[k][0]].frame(index[k][1])
    cfg = RefineConfig(voxel_size=voxel, depth_max=2.0 if subject else 1.0)
    if seam_ref:
        cfg.seam_ref_frames = int(seam_ref)
    print(f"\n정밀 재구성 시작 (프레임 {len(index)}개, 패스 {len(set(passes))}개). 몇 분 걸릴 수 있습니다.")
    try:
        v, f, c, rep = Refiner(K, scale, cfg, subject_mode=subject).run((len(index), get), poses, passes)
    except Exception as e:  # noqa: BLE001 - 실패하면 실시간 결과를 그대로 쓴다
        print(f"정밀 재구성 실패: {e}")
        return None
    if len(f) == 0:
        print("정밀 재구성 결과가 비었습니다.")
        return None
    path = os.path.join(out_dir, "scene.ply")
    prev = os.path.join(out_dir, "scene_prev.ply")
    if os.path.exists(path) and not os.path.exists(prev):
        import shutil
        shutil.copy2(path, prev)
    mt.save_mesh(path, v, f, c)
    print(f"정밀 재구성 장면 저장: {path}  (삼각형 {len(f):,}개)")
    return v, f, c


def _find_recording(path):
    p = os.path.abspath(path)
    if os.path.isdir(p):
        p = os.path.join(p, "frames.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"녹화 파일이 없습니다: {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description="RealSense D435 3D 스캐너")
    ap.add_argument("--open", metavar="PATH",
                    help="저장된 파일을 편집기로 열기: 메시(PLY/OBJ/STL), 녹화(frames.npz → 정밀 재구성), 스캔 폴더")
    ap.add_argument("--load", action="store_true", help="편집기를 열고 파일 선택 창 띄우기")
    ap.add_argument("--repair", metavar="PATH", help="다듬기 도구(조각 잇기·구멍 메우기)로 메시 열기")
    ap.add_argument("--refine", metavar="SCAN", help="저장된 스캔 폴더를 다시 정밀 재구성")
    ap.add_argument("--merge", metavar="SCAN", nargs="+",
                    help="여러 스캔 폴더를 이어 붙여 정밀 재구성 (예: 정면 스캔 + 위에서 찍은 스캔). "
                         "'latest2'면 최근 두 스캔")
    ap.add_argument("--no-refine", action="store_true", help="스캔 후 정밀 재구성을 건너뜀")
    ap.add_argument("--recorrect", metavar="SCAN", nargs="*",
                    help="기존 스캔을 최신 보정(마지막 연결 포함)으로 다시 계산. 폴더를 안 주면 scans 안의 전부")
    ap.add_argument("--seam-ref", type=int, default=10,
                    help="마지막 연결: 끝 장면과 순서대로 비교할 처음 장면 수 N (기본 10)")
    ap.add_argument("--refine-voxel", type=float, default=0.002,
                    help="정밀 재구성 복셀 크기 m (기본 0.002 = 2mm, 작을수록 세밀하지만 느림)")
    ap.add_argument("--voxel", type=float, default=0.004, help="복셀 크기 m (기본 0.004 = 4mm)")
    ap.add_argument("--min-depth", type=float, default=0.15, help="최소 거리 m (기본 0.15)")
    ap.add_argument("--max-depth", type=float, default=1.0, help="최대 거리 m (기본 1.0)")
    ap.add_argument("--subject", action="store_true",
                    help="사람 스캔 모드: 카메라 고정 + 회전의자, 배경 자동 제거")
    ap.add_argument("--preset", default="high_accuracy", choices=["high_accuracy", "default", "high_density"],
                    help="깊이 프리셋 (기본 high_accuracy: 노이즈 적음. 머리카락 구멍이 많으면 default)")
    ap.add_argument("--subject-depth", type=float, default=0.3,
                    help="사람 모드에서 대상 거리 앞뒤로 남길 폭 m (기본 0.3)")
    args = ap.parse_args()

    import mesh_tools as mt

    if args.recorrect is not None:
        from scan import Recording
        targets = args.recorrect or sorted(
            os.path.join(SCANS, d) for d in os.listdir(SCANS)
            if os.path.exists(os.path.join(SCANS, d, "frames.npz")))
        if not targets:
            print("다시 보정할 녹화(frames.npz)가 있는 스캔이 없습니다.")
            return 1
        done = 0
        for t in targets:
            try:
                rec_path = _find_recording(t)
            except FileNotFoundError as e:
                print(f"건너뜀: {e}")
                continue
            print(f"\n===== 다시 보정: {os.path.dirname(rec_path)} =====")
            r = _refine_recordings([Recording.load(rec_path)], args.refine_voxel,
                                   os.path.dirname(rec_path), seam_ref=args.seam_ref)
            done += r is not None
        print(f"\n다시 보정 완료: {done}/{len(targets)}개. 이전 결과는 각 폴더의 scene_prev.ply")
        return 0
    if args.repair:
        from repair import run_repair
        p = os.path.abspath(args.repair)
        if os.path.isdir(p):
            p = os.path.join(p, "scene.ply")
        run_repair(p, default_dir=os.path.dirname(p))
        return 0
    if args.open or args.load:
        from editor import run_editor
        os.makedirs(SCANS, exist_ok=True)
        run_editor(default_dir=SCANS, scans_dir=SCANS, open_path=args.open)
        return 0
    elif args.refine or args.merge:
        from scan import Recording
        srcs = args.merge or [args.refine]
        if srcs == ["latest2"]:
            dirs = sorted(d for d in os.listdir(SCANS)
                          if os.path.exists(os.path.join(SCANS, d, "frames.npz")))
            if len(dirs) < 2:
                print("녹화가 있는 스캔이 2개 이상 필요합니다.")
                return 1
            srcs = [os.path.join(SCANS, d) for d in dirs[-2:]]
        paths = [_find_recording(p) for p in srcs]
        for p in paths:
            print(f"녹화 불러오기: {p}")
        recs = [Recording.load(p) for p in paths]
        if args.merge:
            out_dir = os.path.join(SCANS, datetime.now().strftime("%Y%m%d_%H%M%S") + "_merged")
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = os.path.dirname(paths[0])
        res = _refine_recordings(recs, args.refine_voxel, out_dir)
        if res is None:
            return 1
        v, f, c = res
    else:
        from scan import ScanConfig, run_scan
        cfg = ScanConfig(voxel_size=args.voxel, depth_min=args.min_depth, depth_max=args.max_depth,
                         subject_mode=args.subject, subject_half_depth=args.subject_depth,
                         preset=args.preset)
        if args.subject:
            # 배경은 마스크로 지우므로 거리 제한은 넉넉하게 (대상이 1m 넘게 떨어져 앉아도 되도록)
            cfg.depth_max = max(cfg.depth_max, 2.0)
            # 조명은 고정인데 사람만 돌면 밝기 패턴이 화면에 머물러 컬러 기반 추적이 회전을 덜 잡는다
            # (번짐의 원인). 실시간 미리보기는 형태만 쓰고, 최종 결과는 정밀 재구성이 담당.
            cfg.track_method = "point_to_plane"
        try:
            res = run_scan(cfg)
        except RuntimeError as e:
            print(f"\n오류: {e}")
            return 1
        if res is None:
            print("스캔을 취소했습니다.")
            return 0
        v, f, c = res.mesh
        out_dir = os.path.join(SCANS, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        rec_path = os.path.join(out_dir, "frames.npz")
        res.recording.save(rec_path)
        print(f"녹화 저장: {rec_path}  (프레임 {len(res.recording)}개)")
        if len(f):
            mt.save_mesh(os.path.join(out_dir, "scene_live.ply"), v, f, c)
            print(f"실시간 결과 저장: scene_live.ply  (삼각형 {len(f):,}개)")
        if not args.no_refine:
            refined = _refine_recordings([res.recording], args.refine_voxel, out_dir)
            if refined is not None:
                v, f, c = refined
        if len(f) == 0:
            print("복원된 메시가 비었습니다. 더 천천히, 물체가 화면에 잘 보이게 다시 스캔해 보세요.")
            return 1

    from editor import run_editor
    run_editor(v, f, c, default_dir=out_dir, name=os.path.basename(out_dir), scans_dir=SCANS)
    return 0


def run_cli(argv=None):
    """명령줄 모드로 실행하고 종료 코드를 돌려준다 (런처가 작업마다 별도 프로세스로 부름)."""
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    _start_log()
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    print(f"[{datetime.now():%H:%M:%S}] 종료 (코드 {code}). 로그: {LOG_PATH}")
    return code


if __name__ == "__main__":
    sys.exit(run_cli())
