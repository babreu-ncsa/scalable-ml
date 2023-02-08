from mpi4py import MPI
import subprocess
import json
import os

def get_worker_ip():
    """
    Return a string with the IP address of the host where
    this process is running.
    """
    ip = subprocess.check_output(["hostname", "-I"])
    ip = str(ip)
    ip = ip.replace('b','')
    ip = ip.replace("'", "")
    ip = ip.replace(r" \n", "")
    ip = ip.split(" ")[-1]

    return ip

def set_tf_config():
    """
    Sets TF_CONFIG env variable for each process.
    """

    # start mpi
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    # collect ip of hosts, one mpi process per host
    my_ip = get_worker_ip() + ":8517"
    ips_and_ranks = comm.allgather((my_ip, my_rank))

    # separate chief and workers
    worker_ips = []
    chief_ip = []
    for ip_rank in ips_and_ranks:
        ip, rank = ip_rank
        if rank != 0:
            worker_ips.append(ip)
        else:
            chief_ip.append(ip)

    # set env variable
    if my_rank == 0:
        MY_TF_CONFIG = json.dumps({
            "cluster": {
                "chief": chief_ip,
                "worker": worker_ips,
            },
            "task": {"type": "chief", "index": 0}
        })
    else:
        MY_TF_CONFIG = json.dumps({
            "cluster": {
                "chief": chief_ip,
                "worker": worker_ips,
            },
            "task": {"type": "worker", "index": my_rank-1}
        })

    os.environ["TF_CONFIG"] = MY_TF_CONFIG
    print("Hello from host %d" % my_rank)
    print(MY_TF_CONFIG)
    return

def _is_chief(task_type, task_id):
    """
    We are using a chief, but there are cases where only workers are used.
    """
    return task_type == 'chief'
