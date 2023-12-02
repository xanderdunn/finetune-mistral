import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl",
                        init_method="tcp://10.141.0.12:1234",
                        world_size=2,
                        rank=0)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.Tensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)
print(tensor_list)
