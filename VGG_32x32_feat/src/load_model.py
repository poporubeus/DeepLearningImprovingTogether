import torch
from model import MyVGG
import logging
import torch.optim as optim
from utils import model_save_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = MyVGG()
opt = optim.Adam(model.parameters(), lr=0.001)
checkpoint = torch.load(model_save_PATH+f'weights.pt')
model.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epochs']
loss = checkpoint['training_loss']


logger.info("Model loaded successfully")


for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in opt.state_dict():
    print(var_name, "\t", opt.state_dict())


