import torch
import torch.distributed as dist
from pydantic import ValidationError
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
  master_addr: str                 
  master_port: int = 29500         
  world_size: int                  
  rank: int                       

try:
  config = Settings()
  print(config.master_addr, config.master_port, config.world_size, config.rank)

  dist.init_process_group(
    backend="gloo",
    init_method=f"tcp://{config.master_addr}:{config.master_port}",
    world_size=config.world_size,
    rank=config.rank
  )

  tensor = torch.zeros(1)  
  if config.rank == 0:
    dist.send(tensor, dst=1)
    print(f"Rank {config.rank} sent tensor {tensor.item()} to rank 1")

    dist.recv(tensor, src=config.world_size - 1)
    print(f"Rank {config.rank} received tensor {tensor.item()} from rank {config.world_size - 1}")
  else:
    dist.recv(tensor, src=config.rank - 1)
    print(f"Rank {config.rank} received tensor {tensor.item()} from rank {config.rank - 1}")

    tensor += 1
    next_rank = (config.rank + 1) % config.world_size
    dist.send(tensor, dst=next_rank)
    print(f"Rank {config.rank} sent tensor {tensor.item()} to rank {next_rank}")

    dist.destroy_process_group()

except ValidationError as e:
  print("ERROR:", e)
  print(e)
