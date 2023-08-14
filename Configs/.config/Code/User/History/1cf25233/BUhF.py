import torch
from torch.utils.tensorboard import SummaryWriter
from core.flowNetS import FlowNetS
from torchsummary import summary

writer = SummaryWriter("logs")

dummy_input = torch.randn(2,10,10,2).cuda()
input_shape = dummy_input.shape
model = FlowNetS().cuda()

# add conceptual graph to tensorboard
writer.add_graph(model, dummy_input)
#summary(model, input_size=input_shape)
# do one step
writer.close()

