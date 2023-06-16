from __future__ import print_function
from collections import defaultdict, deque
import datetime
import math
import time
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import errno
import os

import sys


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
                sys.stdout.flush()

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


# def init_distributed_mode(args):
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ['WORLD_SIZE'])
#         print(f"RANK and WORLD_SIZE in environment: {rank}/{world_size}")
#     else:
#         rank = -1
#         world_size = -1

#     torch.cuda.set_device(args.local_rank)
#     torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
#     torch.distributed.barrier()
#     setup_for_distributed(is_main_process())

#     if args.output_dir:
#         mkdir(args.output_dir)
#     if args.model_id:
#         mkdir(os.path.join('./models/', args.model_id))


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

    if args.output_dir:
        mkdir(args.output_dir)
    # if args.model_id:
    #     mkdir(os.path.join('./models/', args.model_id))


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union



import numpy as np



def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


class RefIou:
    def __init__(self, len_thresh):
        self.cum_I = 0
        self.cum_U = 0
        self.eval_seg_iou_list = [.5, .6, .7, .8, .9]
        self.acc_ious = 0
        self.seg_correct = np.zeros(len(self.eval_seg_iou_list), dtype=np.int32)
        self.seg_total = 0

        self.cum_s_I = 0
        self.cum_s_U = 1e-8
        self.acc_ious_s = 0
        self.seg_total_s = 0

        
        self.cum_l_I = 0
        self.cum_l_U = 1e-8
        self.acc_ious_l = 0
        self.seg_total_l = 0

        self.len_thresh = len_thresh

    def update(self, output, target, temp_len):
        iou, I, U = IoU(output, target)
        self.acc_ious += iou
        self.cum_I += I
        self.cum_U += U
        self.seg_total += 1
        for n_eval_iou in range(len(self.eval_seg_iou_list)):
            eval_seg_iou = self.eval_seg_iou_list[n_eval_iou]
            self.seg_correct[n_eval_iou] += (iou >= eval_seg_iou)

        if temp_len > self.len_thresh:
            self.cum_l_U += U
            self.cum_l_I += I
            self.acc_ious_l += iou
            self.seg_total_l += 1
        else:
            self.cum_s_U += U
            self.cum_s_I += I
            self.acc_ious_s += iou
            self.seg_total_s += 1

        
    
    def reduce_from_all_processes(self):
        t = reduce_across_processes([self.acc_ious, self.cum_I, self.cum_U, self.seg_total])
        self.acc_ious, self.cum_I, self.cum_U, self.seg_total = t
        self.seg_correct = reduce_across_processes(self.seg_correct)
        self.over_iou = self.cum_I * 100 / self.cum_U

        t_l = reduce_across_processes([self.acc_ious_l, self.cum_l_I, self.cum_l_U, self.seg_total_l])
        self.acc_ious_l, self.cum_l_I, self.cum_l_U, self.seg_total_l = t_l
        self.over_iou_l = self.cum_l_I * 100 / self.cum_l_U

        t_s = reduce_across_processes([self.acc_ious_s, self.cum_s_I, self.cum_s_U, self.seg_total_s])
        self.acc_ious_s, self.cum_s_I, self.cum_s_U, self.seg_total_s = t_s
        self.over_iou_s = self.cum_s_I * 100 / self.cum_s_U

    def __str__(self):
        results_str = ''
        for n_eval_iou in range(len(self.eval_seg_iou_list)):
            results_str += '    precision@%s = %.2f\n' % \
                        (str(self.eval_seg_iou_list[n_eval_iou]), self.seg_correct[n_eval_iou] * 100. / self.seg_total)
             

        return ('mean iou {:.2f}\n over iou {: .2f}\n').format(self.acc_ious * 100 / self.seg_total, self.cum_I * 100 / self.cum_U) + results_str + \
             ('mean l iou {:.2f}\n over l iou {: .2f}\n').format(self.acc_ious_l * 100 / self.seg_total_l, self.cum_l_I * 100 / self.cum_l_U) + \
              ('mean s iou {:.2f}\n over s iou {: .2f}\n').format(self.acc_ious_s * 100 / self.seg_total_s, self.cum_s_I * 100 / self.cum_s_U)

import warnings
class PolyLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False, min_lr=1e-4):
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, lr_lambda, last_epoch, verbose)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [((base_lr - self.min_lr) * lmbda(self.last_epoch) + self.min_lr)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def get_lr_scheduler(args, optimizer, iters_per_epoch):
    
    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list([iters_per_epoch * i for i in args.lr_steps]), gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= (args.epochs - args.lr_warmup_epochs) * iters_per_epoch, eta_min = 1e-6)
    elif args.lr_scheduler == "polylr":
        main_lr_scheduler = PolyLR(optimizer, lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9, min_lr=args.min_lr)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
        
    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler
        
    return lr_scheduler



@torch.no_grad()        
def _momentum_update(model, model_momentum, momentum): 
    for param, param_m in zip(model.parameters(), model_momentum.parameters()):
        param_m.data = param_m.data * momentum + param.data * (1. - momentum)


from torch.utils.data.distributed import DistributedSampler

class DistributedSampler_LEN(DistributedSampler):
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.__len__()])
    def __len__(self) -> int:
        return 1321 * 8
