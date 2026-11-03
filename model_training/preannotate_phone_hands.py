"""Propose missing hand boxes on phone-source images in datasets/hands-phones.

Workflow:
  1) python model_training/preannotate_phone_hands.py propose
  2) Review datasets/hand_review/review.csv and previews/*.jpg; mark status ACCEPT where correct.
  3) python model_training/preannotate_phone_hands.py apply

Only reviewed ACCEPT labels are written to datasets/hands-phones-v2. Existing phone
labels are preserved. Original datasets never modified. Validation/test labels
are not pseudo-labelled by default. The review is *not* a substitute for checking
images without proposals; those may still have missing annotations.
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT / 'datasets'
SOURCE = DATASETS / 'hands-phones'
TARGET = DATASETS / 'hands-phones-v2'
REVIEW = DATASETS / 'hand_review'
MODEL = PROJECT / 'models' / 'landmarker' / 'hand_landmarker.task'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
HEADERS = ['split', 'image', 'boxes', 'status', 'notes']


def fmt(box):
    return '0 ' + ' '.join(f'{v:.8f}' for v in box)


def to_box(landmarks, margin=0.12):
    coords = np.array([[float(p.x), float(p.y)] for p in landmarks])
    if not np.isfinite(coords).all():
        return None
    x1, y1 = coords.min(axis=0)
    x2, y2 = coords.max(axis=0)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    x1 = max(0., x1 - margin * w)
    y1 = max(0., y1 - margin * h)
    x2 = min(1., x2 + margin * w)
    y2 = min(1., y2 + margin * h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]


def draw_label(image, values, color, caption):
    height, width = image.shape[:2]
    x, y, w, h = map(float, values)
    p1 = (round((x-w/2)*width), round((y-h/2)*height))
    p2 = (round((x+w/2)*width), round((y+h/2)*height))
    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(image, caption, (max(p1[0], 0), max(p1[1]-5, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def propose(args):
    if not MODEL.is_file():
        raise FileNotFoundError(
            f'Missing {MODEL}\nDownload official hand_landmarker.task from:\n'
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/1/hand_landmarker.task')
    if not SOURCE.is_dir():
        raise FileNotFoundError(f'First create your merged dataset: {SOURCE}')
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / 'previews').mkdir(exist_ok=True)
    (REVIEW / 'proposals').mkdir(exist_ok=True)
    if (REVIEW / 'review.csv').exists() and not args.overwrite:
        raise FileExistsError('review.csv exists: refusing to overwrite review decisions. Use --overwrite to regenerate.')

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.confidence,
        min_hand_presence_confidence=args.confidence,
    )
    rows = []
    totals = {'images': 0, 'with_proposals': 0, 'hands': 0, 'unreadable': 0}
    with mp.tasks.vision.HandLandmarker.create_from_options(opts) as detector:
        for split in args.splits:
            img_dir = SOURCE / 'images' / split
            label_dir = SOURCE / 'labels' / split
            if not img_dir.is_dir():
                raise FileNotFoundError(img_dir)
            preview_dir = REVIEW / 'previews' / split
            proposal_dir = REVIEW / 'proposals' / split
            preview_dir.mkdir(parents=True, exist_ok=True)
            proposal_dir.mkdir(parents=True, exist_ok=True)
            paths = sorted(x for x in img_dir.iterdir()
                           if x.name.startswith('phone_') and x.suffix.lower() in IMAGE_SUFFIXES)
            if args.limit:
                paths = paths[:args.limit]
            for index, path in enumerate(paths, 1):
                totals['images'] += 1
                im = cv2.imread(str(path))
                if im is None:
                    totals['unreadable'] += 1
                    continue
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                found = detector.detect(mp_im)
                boxes = [box for hand in found.hand_landmarks if (box := to_box(hand, args.margin)) is not None]
                # Skip images without candidate hand boxes, but do not assume hands are absent.
                if not boxes:
                    continue
                totals['with_proposals'] += 1
                totals['hands'] += len(boxes)
                proposal_file = proposal_dir / f'{path.stem}.txt'
                proposal_file.write_text('\n'.join(map(fmt, boxes)) + '\n', encoding='utf-8')
                for box in boxes:
                    draw_label(im, box, (0, 200, 0), 'HAND - candidate')
                old_label = label_dir / f'{path.stem}.txt'
                if old_label.exists():
                    for line in old_label.read_text(encoding='utf-8-sig').splitlines():
                        tokens = line.split()
                        if len(tokens) == 5 and tokens[0] == '1':
                            draw_label(im, tokens[1:], (0, 0, 255), 'PHONE - original')
                h, w = im.shape[:2]
                if max(h, w) > 1000:
                    scale = 1000 / max(h, w)
                    im = cv2.resize(im, (round(w*scale), round(h*scale)))
                cv2.imwrite(str(preview_dir / f'{path.stem}.jpg'), im)
                rows.append({'split': split, 'image': path.name, 'boxes': len(boxes),
                             'status': 'PENDING', 'notes': ''})
                if index % 250 == 0:
                    print(f'{split}: scanned {index}/{len(paths)} images; proposals {totals["with_proposals"]}')
    with (REVIEW / 'review.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print('Review proposals:', REVIEW / 'review.csv')
    print('Previews:', REVIEW / 'previews')
    print('Summary:', totals)
    print('Mark verified rows ACCEPT, rejected rows REJECT. Leave uncertain rows PENDING.')


def apply(args):
    review_csv = REVIEW / 'review.csv'
    if not review_csv.exists():
        raise FileNotFoundError(f'Run propose first: {review_csv}')
    with review_csv.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    accepted = [r for r in rows if r['status'].strip().upper() == 'ACCEPT']
    print(f'Accepted {len(accepted)} / {len(rows)} proposal images')
    if not accepted:
        raise ValueError('No ACCEPT rows in review.csv. No files written.')
    if TARGET.exists():
        raise FileExistsError(f'{TARGET} already exists. Move or remove it manually before applying.')
    # Stage entire dataset. shutil.copytree keeps source data unchanged.
    print('Copying original merged dataset; this may use substantial disk space...')
    try:
        shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns('*.cache'))
        count = 0
        for row in accepted:
            split = row['split']
            stem = Path(row['image']).stem
            if split not in {'train', 'val', 'test'} or not stem.startswith('phone_'):
                raise ValueError(f'Unexpected review record: {row}')
            original_image = SOURCE / 'images' / split / row['image']
            if not original_image.is_file():
                raise FileNotFoundError(original_image)
            proposal = REVIEW / 'proposals' / split / f'{stem}.txt'
            if not proposal.exists():
                raise FileNotFoundError(proposal)
            target_label = TARGET / 'labels' / split / f'{stem}.txt'
            old_lines = [s.strip() for s in target_label.read_text(encoding='utf-8-sig').splitlines() if s.strip()] if target_label.exists() else []
            new_lines = [s.strip() for s in proposal.read_text().splitlines() if s.strip()]
            for line in new_lines:
                parts = line.split()
                if len(parts) != 5 or parts[0] != '0':
                    raise ValueError(f'Invalid proposal: {proposal}')
                coords = [float(x) for x in parts[1:]]
                if not all(np.isfinite(coords)) or not all(0 <= x <= 1 for x in coords) or coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError(f'Invalid coordinates: {proposal}')
            target_label.write_text('\n'.join(old_lines + new_lines) + '\n', encoding='utf-8')
            count += len(new_lines)
        yaml = TARGET / 'data.yaml'
        if yaml.exists():
            lines = yaml.read_text(encoding='utf-8').splitlines()
            lines = [('path: ' + TARGET.as_posix()) if line.strip().startswith('path:') else line for line in lines]
            yaml.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'Done. Added {count} hand boxes on {len(accepted)} images in {TARGET}')
        print('IMPORTANT: still audit PENDING/no-proposal images and unlabelled phones in hand-source images.')
        print('Validation metrics against original partial annotations are NOT a reliable two-class evaluation.')
    except Exception:
        print(f'Error during apply. Partial output may exist at {TARGET}; inspect/remove before retrying.')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('propose', help='Generate candidate boxes and review previews')
    p.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train'],
                   help='Default train only; validation/test annotations need independent manual audit')
    p.add_argument('--limit', type=int, default=0, help='Optional sample size per split for quick trial')
    p.add_argument('--max-hands', type=int, default=4)
    p.add_argument('--confidence', type=float, default=0.35)
    p.add_argument('--margin', type=float, default=0.20,
                   help='Relative margin around landmark extrema; manually verify full-hand extent')
    p.add_argument('--overwrite', action='store_true')
    sub.add_parser('apply', help='Copy merged dataset to v2 and add reviewed ACCEPT hand boxes')
    args = parser.parse_args()
    (propose if args.command == 'propose' else apply)(args)


if __name__ == '__main__':
    main()
