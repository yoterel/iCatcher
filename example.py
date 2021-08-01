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
    parser.add_argument("-m", "--model", type=str, choices=["icatcher", "icatcher+"], default="icatcher", help="which model will be used for predictions")
    # Set up text output file, using https://osf.io/3n97m/ - PrefLookTimestamp coding standard
    parser.add_argument('--output_format', type=str, default="PrefLookTimestamp", choices=["PrefLookTimestamp"])
    parser.add_argument('--output_video_path', help='if present, annotated video will be saved to this path')
    parser.add_argument('--show_output', action='store_true', help='show results online in a separate window')
    parser.add_argument('--per_channel_mean', nargs=3, metavar=('Channel1_mean', 'Channel2_mean', 'Channel3_mean'),
                        type=float, help='supply custom per-channel mean of data for normalization')
    parser.add_argument('--per_channel_std', nargs=3, metavar=('Channel1_std', 'Channel2_std', 'Channel3_std'),
                        type=float, help='supply custom per-channel std of data for normalization')
    parser.add_argument('--gpu_id', type=int, default=-1, help='GPU id to use, use -1 for CPU.')
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
    if not args.per_channel_mean:
        if args.model == "icatcher":
            args.per_channel_mean = [0.41304266, 0.34594961, 0.27693587]
        elif args.model == "icatcher+":
            args.per_channel_mean = [0.485, 0.456, 0.406]
        else:
            raise NotImplementedError
    if not args.per_channel_std:
        if args.model == "icatcher":
            args.per_channel_std = [0.28606387, 0.2466201, 0.20393684]
        elif args.model == "icatcher+":
            args.per_channel_std = [0.229, 0.224, 0.225]
        else:
            raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_arguments()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    args.gpu_id = utils.configure_compute_environment(args.gpu_id, args.model)
    predict.predict_from_video(args)
