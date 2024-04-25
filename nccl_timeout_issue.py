import torch.distributed as dist
import os
from datetime import timedelta
import time

_sleep_time = 25
_timeout_delta = 15
_comm_framework = 'nccl'  # 'gloo' or 'nccl'
_start_barrier: bool = True  # False reproes timeout failure, True enables expected timeout

def main(sleep_rank, sleep_barrier):
    print(f"entering main {sleep_rank=}, {sleep_barrier=}")

    dist.init_process_group(_comm_framework, rank=int(os.environ['RANK']), world_size=2, timeout=timedelta(seconds=_timeout_delta))

    _rank = dist.get_rank()
    print(f"Past dist init {_rank=}")
    if _start_barrier:
        dist.barrier()

    if _rank == sleep_rank and sleep_barrier == 0:
        print(f"Sleeping {_rank=}")
        time.sleep(_sleep_time)
    dist.barrier()
    print(f"Past first barrier, {_rank=}")

    if _rank == sleep_rank and sleep_barrier == 1:
        print(f"Sleeping {_rank=}")
        time.sleep(_sleep_time)
    print(f"At second barrier, {_rank=}")
    dist.barrier()
    print(f"Past second barrier. {_rank=}")


if __name__ == '__main__':
    import sys
    sleep_rank = int(sys.argv[1])
    sleep_barrier = int(sys.argv[2])
    main(sleep_rank, sleep_barrier)
