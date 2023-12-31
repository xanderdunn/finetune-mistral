# Finetune Mistral
Finetuning Mistral on Multi-Node GPUs.

## Setup
- Assumes CUDA is already setup
- Assumes torch 2.0.1 is already installed
- `python -c "import torch; print(torch.version.cuda)"`, expect it to say 11.7 given the version of pytorch.
- `python3 -m pip install --upgrade pip`
- `pip3 install transformers accelerate deepspeed trl pyparsing`
- Set the ulimit for locked memory to unlimited
    - `sudo vim /etc/ssh/sshd_config`
    - Make sure this is set to yes: `UsePAM yes`
    - `sudo service sshd restart`
    - `sudo vim /etc/security/limits.conf` and add:
    ```
    * soft memlock unlimited
    * hard memlock unlimited
    ```
    - log out
    - log back in
    - Now `ulimit -l` should be unlimited


## Test Cross-Node Comms
- Test Infiniband:
    - On machine 0: `ib_write_bw -a --report_gbits`
    - On machine 1: `ib_write_bw -a --report_gbits 10.141.0.12`
    - It should finish with something like this:
    ```
     #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
     8388608    5000             195.77             195.77 		   0.002917
    ```
- Test torch distributed:
    - On machine 0: Run `MASTER_ADDR=10.141.0.12 MASTER_PORT=1234 python3 test.py`
    - On machine 1: Modify test.py `rank=1` and then run `MASTER_ADDR=10.141.0.12 MASTER_PORT=1234 python3 test.py`
    - If it fails, setting `export NCCL_DEBUG=info` can provide some additional information.
    - On success, you should see:
    ```
    [tensor([16.], device='cuda:0'), tensor([16.], device='cuda:1'), tensor([16.], device='cuda:2'), tensor([16.], device='cuda:3'), tensor([16.], device='cuda:4'), tensor([16.], device='cuda:5'), tensor([16.], device='cuda:6'), tensor([16.], device='cuda:7')]
    ```
