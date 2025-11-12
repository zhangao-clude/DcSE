import argparse
import os


def settings():
    parser = argparse.ArgumentParser()

    # DenseNet parameters
    parser.add_argument('--growth_rate', type=int, default=32,
                        help='growth_rate. Default is 32.')

    parser.add_argument('--layer_per_block', type=int, default=5,
                        help='layer_per_block is 5.')

    parser.add_argument('--num_blocks', type=int, default=2,
                        help='num_blocks is 2.')

    parser.add_argument('--densenet_dropout', type=float, default=0.3,
                        help='densenet_dropout. Default is 0.3.')

    # Training settings
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate. Default is 1e-4.')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout. Default is 0.2.')

    parser.add_argument('--weight_decay', default=5e-4,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size. Default is 64.')

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of epochs to train. Default is 30.')

    parser.add_argument('--k_folds', type=int, default=5,
                        help='k_folds. Default is 5')

    # CBAM DcSEResult parameter setting
    parser.add_argument('--reduction', type=int, default=16,
                        help='reduction. Default is 16')

    parser.add_argument('--spatial_kernel', type=int, default=9,
                        help='spatial_kernel. Default is 9')

    # Encoder settings
    parser.add_argument('--nhead', type=int, default=4,
                        help='nhead. Default is 4')

    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='spatial_kernel. Default is 9')

    args = parser.parse_args()

    return args
