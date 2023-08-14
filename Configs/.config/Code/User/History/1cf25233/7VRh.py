import torch
from torch.utils.tensorboard import SummaryWriter
from core.flowNetS import FlowNetS

writer = SummaryWriter("logs/deleteme")

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
    writer.add_scalar("Loss/train_epoch", loss, i)
    



# add conceptual graph to tensorboard
writer.add_graph(model, dummy_input)
# do one step
writer.close()

