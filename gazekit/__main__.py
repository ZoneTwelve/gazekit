"""CLI: python -m gazekit <command>"""

import argparse


def main():
    p = argparse.ArgumentParser(prog="gazekit",
                                description="Webcam eye tracking toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cameras", help="list available cameras")

    au = sub.add_parser("auto",
                        help="guided end-to-end: collect everything missing, "
                             "train, evaluate, then adapt in ambient mode")
    au.add_argument("--camera", type=int, default=0)
    au.add_argument("--full", action="store_true",
                    help="run every collection step even if data exists")
    au.add_argument("--cnn", action="store_true",
                    help="force CNN training")
    au.add_argument("--no-cnn", action="store_true")
    au.add_argument("--no-ambient", action="store_true",
                    help="don't start ambient mode at the end")

    d = sub.add_parser("doctor", help="run the environment check only")
    d.add_argument("--camera", type=int, default=0)

    c = sub.add_parser("calibrate", help="run the full training process")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--points", type=int, default=16, choices=(9, 13, 16),
                   help="grid size (16 = 4x4, recommended)")
    c.add_argument("--rounds", type=int, default=2,
                   help="passes over the grid (2 recommended)")
    c.add_argument("--out", default="data/gaze_model.pkl")

    co = sub.add_parser("collect", help="extra training-data scenarios")
    co.add_argument("scenario",
                    choices=("pursuit", "edges", "posture", "vor", "blinks"))
    co.add_argument("--camera", type=int, default=0)

    l = sub.add_parser("live", help="show the live gaze dot")
    l.add_argument("--camera", type=int, default=0)
    l.add_argument("--backend", choices=("ridge", "cnn", "hybrid"),
                   default="ridge")
    l.add_argument("--model", default=None)
    l.add_argument("--no-align", action="store_true",
                   help="skip the 3-point quick alignment at start")

    a = sub.add_parser("ambient",
                       help="background trainer: popup dots while you work")
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--min-wait", type=float, default=15.0,
                   help="seconds between popups, lower bound")
    a.add_argument("--max-wait", type=float, default=45.0,
                   help="seconds between popups, upper bound")
    a.add_argument("--quiet", action="store_true",
                   help="no voice cue when a dot appears")
    a.add_argument("--test", action="store_true",
                   help="6s overlay visibility check, no camera needed")

    v = sub.add_parser("verify",
                       help="mouse-as-ground-truth error measurement")
    v.add_argument("--camera", type=int, default=0)
    v.add_argument("--mode", choices=("free", "path"), default="free",
                   help="free = roam anywhere; path = follow a wide track")
    v.add_argument("--teach", action="store_true",
                   help="also save samples for training (tag: mouse)")

    it = sub.add_parser("iterate",
                        help="dataset lifecycle: clean/train/validate/"
                             "evaluate/update")
    it.add_argument("--no-clean", action="store_true")
    it.add_argument("--no-update", action="store_true")
    it.add_argument("--cnn", action="store_true",
                    help="also train the CNN on the cleaned dataset")

    t = sub.add_parser("train-cnn",
                       help="post-train MobileNetV2 on your calibration dataset")
    t.add_argument("--dataset", default="data/dataset")
    t.add_argument("--out", default="data/gaze_cnn.pt")
    t.add_argument("--epochs", type=int, default=30)

    args = p.parse_args()

    if args.cmd == "auto":
        from .auto import run
        run(camera_index=args.camera, full=args.full,
            cnn="yes" if args.cnn else "no" if args.no_cnn else "auto",
            ambient_after=not args.no_ambient)

    elif args.cmd == "cameras":
        from .camera import list_cameras
        cams = list_cameras()
        if not cams:
            print("no cameras found (check camera permission for your terminal)")
        for c_ in cams:
            print(f"[{c_['index']}] {c_['name']}  {c_['resolution']}")

    elif args.cmd == "doctor":
        import cv2
        from . import ui
        from .calibrate import environment_gate, Aborted
        from .camera import open_camera
        from .screen import screen_size
        from .tracker import FaceTracker
        win = ui.FullscreenWindow("gazekit-doctor", screen_size())
        cap = open_camera(args.camera)
        tracker = FaceTracker("models/face_landmarker.task")
        try:
            environment_gate(win, cap, tracker)
            print("environment check PASSED")
        except Aborted:
            print("aborted")
        finally:
            tracker.close()
            cap.release()
            cv2.destroyAllWindows()

    elif args.cmd == "calibrate":
        from .calibrate import run
        report = run(camera_index=args.camera, points=args.points,
                     rounds=args.rounds, model_out=args.out)
        if report:
            print(f"verdict: {report['verdict']}  "
                  f"mean error {report['mean_error_px']}px "
                  f"({100 * report['mean_error_frac_diag']:.1f}% of diagonal)")

    elif args.cmd == "collect":
        from .collect import run
        run(args.scenario, camera_index=args.camera)

    elif args.cmd == "live":
        from .live import run
        run(camera_index=args.camera, backend=args.backend,
            model_path=args.model, align=not args.no_align)

    elif args.cmd == "ambient":
        if args.test:
            from .ambient import overlay_test
            overlay_test()
        else:
            from .ambient import run
            run(camera_index=args.camera,
                interval=(args.min_wait, args.max_wait),
                voice=not args.quiet)

    elif args.cmd == "verify":
        from .verify import run
        run(camera_index=args.camera, mode=args.mode, teach=args.teach)

    elif args.cmd == "iterate":
        from .evaluate import run
        run(do_clean=not args.no_clean, do_update=not args.no_update,
            train_cnn=args.cnn)

    elif args.cmd == "train-cnn":
        from .cnn import train
        train(dataset_root=args.dataset, out=args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
