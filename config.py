from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description="Softmax loss classification")

# Data
parser.add_argument('--train_data_dir', nargs='+', type=str, metavar='PATH', default=[None])
parser.add_argument('--train_data_gt', nargs='+', type=str, metavar='PATH', default=[None])
parser.add_argument('--test_data_dir', type=str, metavar='PATH', default=None)
parser.add_argument('--test_data_gt', type=str, metavar='PATH', default=None)
parser.add_argument('-b', '--train_batch_size', type=int, default=256)
parser.add_argument('-v', '--val_batch_size', type=int, default=256)
parser.add_argument('-j', '--workers', type=int, default=2)
parser.add_argument('-g', '--gpus', type=str, default='0')
parser.add_argument('--height', type=int, default=64, help="input height")
parser.add_argument('--width', type=int, default=256, help="input width")
parser.add_argument('--aug', type=bool, default=False, help="using data augmentation or not, Note: we use the offline augmented images")
parser.add_argument('--keep_ratio', action='store_true', default=False, help='keep the image ratio of height and width.')
parser.add_argument('--voc_type', type=str, default='LOWERCASE', choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'])
parser.add_argument('--num_train', type=int, default=-1)
parser.add_argument('--num_test', type=int, default=-1)

# Model
parser.add_argument('--max_len', type=int, default=25)
parser.add_argument('--num_block', type=int, default=3)
parser.add_argument('--att_dim', type=int, default=512)
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--num_decoder', type=int, default=5)
parser.add_argument('--hidden_units', type=int, default=1024)

# Optimizer
parser.add_argument('--lr', type=float, default=1.0, # 0.001
                    help="learning rate of new parameters, for pretrained ")
parser.add_argument('--weight_decay', type=float, default=0.9) # the model maybe under-fitting, 0.0 gives much better results.
# parser.add_argument('--decay_iter', type=int, default=100000)
parser.add_argument('--decay_bound', nargs='+', type=int, default=[800000])
parser.add_argument('--lr_stage', nargs='+', type=float, default=[0.0001, 0.00001]) # adadelta: 1.0 0.1 0.01
parser.add_argument('--decay_end', type=float, default=0.00001)
parser.add_argument('--grad_clip', type=float, default=-1.0)
parser.add_argument('--iters', type=int, default=1000000)
parser.add_argument('--decode_type', type=str, default='greed')

parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--pretrained', type=str, default='', metavar='PATH')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', metavar='PATH')
parser.add_argument('--log_iter', type=int, default=100)
parser.add_argument('--summary_iter', type=int, default=1000)
parser.add_argument('--eval_iter', type=int, default=2000)
parser.add_argument('--save_iter', type=int, default=2000)
parser.add_argument('--vis_dir', type=str, metavar='PATH')
parser.add_argument('--metrics_type', type=str, default='accuracy', choices=['accuracy', 'editdistance'])

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args