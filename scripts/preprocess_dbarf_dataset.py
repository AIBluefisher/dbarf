import os
import argparse
from pathlib import Path

from dbarf.geometry.track import load_track_elements, TrackBuilder
from scripts import extract_relative_poses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, default='datasets',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_matches', type=int, default=30,
                        help='Number of image pairs for loc, default: %(default)s')
    parser.add_argument('--disambiguate', type=bool, default=False,
                        help='Enable/Disable disambiguating wrong matches.')
    parser.add_argument('--min_track_length', type=int, default=3)
    parser.add_argument('--max_track_length', type=int, default=40)
    parser.add_argument('--track_degree', type=int, default=3)
    parser.add_argument('--coverage_thres', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--minimal_views', type=int, default=5)
    parser.add_argument('--ds', type=str,
                        choices=['dict', 'smallarray', 'largearray'],
                        default='largearray')
    parser.add_argument('--filter_type', type=str, choices=[
                        'threshold', 'knn', 'mst_min', 'mst_mean', 'percentile'],
                        default='threshold')
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--percentile', type=float)
    parser.add_argument('--colmap_path', type=Path, default='colmap')
    parser.add_argument('--geometric_verification_type',
                        type=str,
                        choices=['default', 'strict'],
                        default='default')
    parser.add_argument('--recon', type=bool, default=False,
                        help='Indicates whether to reconstruct the scene.')
    parser.add_argument('--visualize', type=bool, default=False,
                        help='Whether to visualize the reconstruction.')
    parser.add_argument('--gpu_idx', type=str, default='0')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Extracting relative poses and store as g2o file.
    view_graph_path, database_path, num_view_pairs = extract_relative_poses.main(args=args)

    # Extracting tracks from colmap database.
    track_elements, track_element_pairs = \
        load_track_elements(database_path=database_path)
    track_builder = TrackBuilder(args.min_track_length, args.max_track_length)
    track_builder.build(track_elements, track_element_pairs)
    track_builder.filter()
    track_filename = os.path.join(args.outputs, 'track.txt')
    track_builder.write_to_file(track_filename)
    print(f'Tracks are written to: {track_filename}')


if __name__ == '__main__':
    main()
