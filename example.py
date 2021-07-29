import utils
import argparse
from pathlib import Path
import predict
import logging


def parse_arguments():
    parser = argparse.ArgumentParser(description='Baby Eye Tracker')
    parser.add_argument('source', type=str, help='the source to use (path to video file or webcam id).')
    parser.add_argument('--source_type', type=str, default='file', choices=['file', 'webcam'],
                        help='selects source of stream to use.')
    parser.add_argument('--output_annotation', type=str, help='filename for text output')
    # Set up text output file, using https://osf.io/3n97m/ - PrefLookTimestamp coding standard
    parser.add_argument('--output_format', type=str, default="PrefLookTimestamp", choices=["PrefLookTimestamp"])
    parser.add_argument('--output_video_path', help='if present, annotated video will be saved to this path')
    parser.add_argument('--show_output', action='store_true', help='show results online in a separate window')
    parser.add_argument('--per_channel_mean', nargs=3, metavar=('Channel1_mean', 'Channel2_mean', 'Channel3_mean'),
                        type=float, help='supply custom per-channel mean of data for normalization')
    parser.add_argument('--per_channel_std', nargs=3, metavar=('Channel1_std', 'Channel2_std', 'Channel3_std'),
                        type=float, help='supply custom per-channel std of data for normalization')
    parser.add_argument('--gpu_id', type=str, default='-1', help='GPU id to use, use -1 for CPU.')
    parser.add_argument("--log",
                        help="If present, writes training log to this path")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info",
                        help="Selects verbosity level")
    args = parser.parse_args()
    if args.output_annotation:
        args.output_filepath = Path(args.output_annotation)
    if args.output_video_path:
        args.output_video_path = Path(args.output_video_path)
    if args.log:
        args.log = Path(args.log)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    utils.configure_environment(args.gpu_id)
    predict.predict_from_video(args)
