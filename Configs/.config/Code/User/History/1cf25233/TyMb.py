import torch
from torch.utils.tensorboard import SummaryWriter
from core.flowNetS import FlowNetS
writer = SummaryWriter("logs")

dummy_input = torch.randn(2,2,2, 2).cuda()
model = FlowNetS().cuda()
writer.add_graph(model, dummy_input)
# do one step
writer.close()

