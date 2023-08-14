import torch
from torch.utils.tensorboard import SummaryWriter
from core.flowNetS import FlowNetS
from torchsummary import summary

writer = SummaryWriter("logs")

dummy_input = torch.randn(2,2,10,10).cuda()
input_shape = dummy_input.shape
model = FlowNetS().cuda()
# do 10 steps
for i in range(10):
    output = model(dummy_input)
    writer.add_graph(model, dummy_input)
    print(output.shape)
    print(output)
    loss = output.sum()
    loss.backward()

    

# add conceptual graph to tensorboard
writer.add_graph(model, dummy_input)
summary(model, input_size=input_shape)
# do one step
writer.close()

