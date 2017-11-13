import argparse
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from convert import convert_mat
from render import render_mat
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_dir', type=str, default=None, help='Defaults to directory of input file')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output_dir = args.output_dir or os.path.dirname(args.input_file)
    logger.info('Running with args %s', vars(args))

    with open(args.input_file, 'rb') as input_file:
        data = pickle.load(input_file)

    for imagepath, outputs in data['activations'].items():
        logger.info('Processing %s', imagepath)
        mat = outputs['step2.43']
        mat = convert_mat(mat)
        render_mat(mat, args.output_dir, imagepath, 'png')
