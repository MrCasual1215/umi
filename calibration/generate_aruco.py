#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2


ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def get_dictionary(dict_name: str):
    if dict_name not in ARUCO_DICTS:
        raise ValueError(
            f"Unsupported dictionary: {dict_name}. "
            f"Available: {', '.join(sorted(ARUCO_DICTS))}"
        )
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])


def draw_marker(dictionary, marker_id: int, size: int, border_bits: int):
    img = 255 * (cv2.UMat(size, size, cv2.CV_8UC1).get() * 0 + 1)
    if hasattr(cv2.aruco, "generateImageMarker"):
        cv2.aruco.generateImageMarker(dictionary, marker_id, size, img, border_bits)
    else:
        img = cv2.aruco.drawMarker(dictionary, marker_id, size, borderBits=border_bits)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ArUco marker image(s).")
    parser.add_argument(
        "--dict",
        default="DICT_4X4_50",
        help="ArUco dictionary name.",
    )
    parser.add_argument(
        "--id",
        type=int,
        nargs="+",
        default=[4, 5],
        help="One or more ArUco marker IDs to generate.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=600,
        help="Output marker image size in pixels.",
    )
    parser.add_argument(
        "--border-bits",
        type=int,
        default=1,
        help="Marker border width in bits.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for generated images.",
    )
    parser.add_argument(
        "--prefix",
        default="aruco",
        help="Filename prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dictionary = get_dictionary(args.dict)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_markers = dictionary.bytesList.shape[0]
    for marker_id in args.id:
        if marker_id < 0 or marker_id >= max_markers:
            raise ValueError(
                f"Marker ID {marker_id} is out of range for {args.dict}. "
                f"Valid range: 0 to {max_markers - 1}"
            )

        image = draw_marker(
            dictionary=dictionary,
            marker_id=marker_id,
            size=args.size,
            border_bits=args.border_bits,
        )
        out_path = output_dir / f"{args.prefix}_{args.dict.lower()}_{marker_id}.png"
        ok = cv2.imwrite(str(out_path), image)
        if not ok:
            raise RuntimeError(f"Failed to write image: {out_path}")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
