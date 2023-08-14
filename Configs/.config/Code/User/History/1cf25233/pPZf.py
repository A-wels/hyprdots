import torch
from torch.utils.tensorboard import SummaryWriter
from core.flowNetS import FlowNetS
writer = SummaryWriter("logs")

dummy_input = torch.randn(10,10,2, 2)
model = FlowNetS()
writer.add_graph(model, dummy_input)
# do one step
writer.close()

