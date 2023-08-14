import torch
from torch.utils.tensorboard import SummaryWriter
from core.flowNetS import FlowNetS
writer = SummaryWriter("logs")

dummy_input = torch.randn(10,10)
model = FlowNetS().cuda()
writer.add_graph(model, dummy_input)