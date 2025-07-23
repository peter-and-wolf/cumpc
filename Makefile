run:
	python -m torch.distributed.run --nproc_per_node=3 --master_addr 127.0.0.1 --master_port 29500 node.py