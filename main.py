import argparse
import torch
from train_simpler_cl import train_simpler_cl

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='./experiments', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--dataset', default='Math23k', type=str, choices=['Math23k', 'AsDiv-A'])
    parser.add_argument('--CL', default='SimplerCL', type=str, choices=['SimplerCL', 'TextualCL', 'NoCL'])
    parser.add_argument('--similarity', default='TLWD', type=str, choices=['TLWD', 'TED'])
    parser.add_argument('--H', action='store_true', help='CL from Holistic View')
    parser.add_argument('--P', action='store_true', help='CL from Primary View')
    parser.add_argument('--L', action='store_true', help='CL from Longest View')

    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--vis', action='store_true', help='Save embeddings for TSNE-Visulization')

    parser.add_argument('--ckpt', default='./ckpts/models_best', type=str)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    if args.train:
        if args.CL == 'SimplerCL':
            train_simpler_cl(args)
        elif args.CL == 'TextualCL':


    # elif args.test:
    #     print('Test Begin')
    #     test(args)