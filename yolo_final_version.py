from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import List
import torch, yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
from tqdm import tqdm


# -------------------- helpers --------------------
def fps_from_speed(s: dict):
    ms = s.get("inference", 0) + s.get("postprocess", 0)
    return 1000. / ms if ms else 0.


def _to_float(v):
    if callable(v):
        v = v()
    if hasattr(v, "item"):
        v = v.item()
    return float(v)


def fmt(tag, m):
    mp50 = _to_float(getattr(m.box, "map50", lambda: m.box.map50()))
    prec = _to_float(getattr(m.box, "mp", lambda: m.box.mp()))
    recall = _to_float(getattr(m.box, "mr", lambda: m.box.mr()))
    return (f"{tag:<12}| mAP@0.5 {mp50:6.3f} | "
            f"P {prec:6.3f} | R {recall:6.3f} | FPS {fps_from_speed(m.speed):6.1f}")


def evaluate(model: YOLO, data: Path, imgsz: int, batch: int, tta=False):
    return model.val(data=str(data), imgsz=imgsz, batch=batch, augment=tta,
                     iou=0.7, conf=0.001, workers=0, verbose=False, save=False)


# ----------------- pseudo-labels -----------------
def generate_pseudo_labels(
        model: YOLO,
        images_dir: Path,
        labels_dir: Path,
        imgsz: int = 640,
        conf_thr: float = 0.7,
        max_det: int = 300,
        allow: set[int] | None = None,
        overwrite: bool = False
) -> int:
    """Generate YOLO-format txt pseudo-labels."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs: List[Path] = sorted(p for p in images_dir.rglob('*')
                              if p.suffix.lower() in exts)
    if not imgs:
        LOGGER.warning(f"No images in {images_dir}")
        return 0

    created = 0
    for img in tqdm(imgs, desc="Pseudo-labeling"):
        txt = labels_dir / f"{img.stem}.txt"
        if txt.exists() and not overwrite:
            continue

        pred = model.predict(img, imgsz=imgsz, conf=conf_thr, iou=0.7,
                             max_det=max_det, verbose=False, save=False)[0]
        lines = []
        for b in pred.boxes:
            cls = int(b.cls)
            if allow and cls not in allow:
                continue
            x, y, w, h = b.xywhn[0]
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        if not lines:
            continue

        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(''.join(lines))
        created += 1

    LOGGER.info(colorstr("green", "bold",
                f"[Pseudo-Label] {created} label files written"))
    return created


# --------------------- train ---------------------
def train_once(weights: Path, data_yaml: Path, epochs: int, imgsz: int, batch: int,
               freeze: int, lr0: float, mosaic: float, copy_paste: float,
               auto_aug: str, multi_scale: bool, out_dir: Path,
               amp: bool, patience: int) -> Path:
    model = YOLO(str(weights))
    model.train(data=str(data_yaml), imgsz=imgsz, batch=batch, epochs=epochs,
                freeze=freeze, lr0=lr0, mosaic=mosaic, copy_paste=copy_paste,
                auto_augment=auto_aug, multi_scale=multi_scale, amp=amp,
                cos_lr=True, close_mosaic=10, deterministic=True, workers=0,
                patience=patience, project=str(out_dir), name="train", verbose=True)
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    LOGGER.info(colorstr("green", "bold", f"[Fine-Tune] best ➜ {best}"))
    return best


# -------------------- yaml util ------------------
def dump_yaml(p: Path, c: dict):
    p.write_text(yaml.safe_dump(c, sort_keys=False))


# -------------------- pipeline -------------------
def main(opt):
    t0 = time.time()
    LOGGER.info(colorstr('blue', 'bold',
                f"Running on {'cuda' if torch.cuda.is_available() else 'cpu'}"))

    ds = Path(opt.dataset_dir).resolve()
    imgs_un = ds / "images/unlabeled"
    lbls_un = ds / "labels/unlabeled"

    # ---------- build data yaml ----------
    fine_yaml = ds / "fine_tune_auto.yaml"
    dump_yaml(fine_yaml, {"path": str(ds),
                          "train": "images/train",
                          "val": "images/val",
                          "nc": opt.num_classes,
                          "names": opt.names})

    # ---------- baseline ----------
    base = YOLO(opt.weights)
    base.fuse()
    LOGGER.info(colorstr('blue', 'bold', "\n========== Baseline =========="))
    LOGGER.info(fmt("Baseline", evaluate(base, fine_yaml, opt.imgsz, opt.batch)))
    LOGGER.info(colorstr('blue', 'bold', "\n========== TTA =========="))
    LOGGER.info(fmt("TTA", evaluate(base, fine_yaml, opt.imgsz,
                                    opt.batch, tta=True)))

    # ---------- round-1 fine-tune ----------
    best1 = train_once(opt.weights, fine_yaml, opt.epochs, opt.imgsz,
                       opt.batch, opt.freeze, opt.lr0, opt.mosaic,
                       opt.copy_paste, opt.auto_aug, opt.multi_scale,
                       Path("runs/exp_finetune"), opt.amp, patience=100)

    # ---------- generate / regenerate pseudo-labels ----------
    if imgs_un.exists():
        LOGGER.info(colorstr('cyan', 'bold',
                    "\n===== Generating Pseudo-Labels with fine-tuned model ====="))
        fine_model = YOLO(best1)
        fine_model.fuse() 
        generate_pseudo_labels(fine_model, imgs_un, lbls_un,
                               imgsz=opt.imgsz, conf_thr=opt.pl_conf,
                               allow=set(range(opt.num_classes)),
                               overwrite=opt.regen_pl)

    if opt.only_pl:
        LOGGER.info(colorstr('green', 'bold',
                    "--only_pl set, Pseudo-label generation completed"))
        return

    # ---------- build self-training yaml ----------
    self_yaml = ds / "selftrain_auto.yaml"
    dump_yaml(self_yaml, {"path": str(ds),
                          "train": ["images/train", "images/unlabeled"],
                          "val": "images/val",
                          "nc": opt.num_classes,
                          "names": opt.names})

    # ---------- round-2 self-training ----------
    best2 = train_once(best1, self_yaml, opt.self_epochs, opt.imgsz,
                       opt.batch, opt.freeze_self, opt.lr0, opt.mosaic,
                       opt.copy_paste, opt.auto_aug, opt.multi_scale,
                       Path("runs/exp_iter1"), opt.amp, patience=10)

    # ---------- final evaluation ----------
    tuned = YOLO(best2)
    tuned.fuse()
    LOGGER.info(colorstr('blue', 'bold', "\n========== Fine-Tuned =========="))
    LOGGER.info(fmt("Fine-Tuned", evaluate(tuned, fine_yaml,
                                           opt.imgsz, opt.batch)))
    LOGGER.info(colorstr('green', 'bold',
                f"\nPipeline finished in {(time.time() - t0)/60:.1f} min"))


# -------------------- CLI ------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Auto-YOLOv5 adaptation pipeline")
    ap.add_argument("--dataset_dir", default="/G006/Dataset")
    ap.add_argument("--weights", default="yolov5su.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--self_epochs", type=int, default=40)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--freeze_self", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--copy_paste", type=float, default=0.2)
    ap.add_argument("--auto_aug", default="randaugment")
    ap.add_argument("--multi_scale", action="store_true")
    ap.add_argument("--pl_conf", type=float, default=0.7)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--only_pl", action="store_true",
                    help="Only generate pseudo labels without subsequent training")
    ap.add_argument("--regen_pl", action="store_true",
                    help="If the pseudo-label already exists, it is forced to be regenerated")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--names", nargs="+",
                    default=["Building", "Street Light",
                             "Tree", "Car", "People"])
    opt, _ = ap.parse_known_args()

    assert Path(opt.weights).exists(), f"Checkpoint not found: {opt.weights}"
    for split in ("images/train", "labels/train"):
        assert (Path(opt.dataset_dir) / split).exists(), f"{split} missing"

    main(opt)
