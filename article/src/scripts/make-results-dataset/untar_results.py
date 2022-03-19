#!/usr/bin/env python
# coding: utf-8
"""
unpacks .gz.tar files from ./article/results/tars
into the appropriate sub-directories of ./article
"""
import argparse
import shutil

import pyprojroot


RESULTS_ROOT = pyprojroot.here() / 'results'
TARS_ROOT = RESULTS_ROOT / 'tars'


tars = sorted(TARS_ROOT.glob('*.tar.gz'))


def main(dry_run=True):
    for tar_path in tars:
        print(
            f"\nunpacking: {tar_path}"
        )

        unpacked_path = '/'.join(tar_path.name.replace('.tar.gz', '').split('-'))
        if unpacked_path.startswith('Bengalese_Finches') or unpacked_path.startswith('Canaries'):
            # results directories. Special case to avoid uploading giant .tar.gz files again
            extract_dir = RESULTS_ROOT / unpacked_path
        else:
            extract_dir = pyprojroot.here() / unpacked_path

        print(
            f"To ``extract_dir``: {extract_dir}"
        )
        if not dry_run:
            print("making directory")
            extract_dir.mkdir(exist_ok=True, parents=True)

        if not dry_run:
            shutil.unpack_archive(
                filename=tar_path,
                extract_dir=extract_dir,
                format="gztar"
            )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dry-run', action='store_true'
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(dry_run=args.dry_run)
