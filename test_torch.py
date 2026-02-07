import torch
import time
import tqdm
print(torch.__version__)

# Check if CUDA is available
if True: #torch.cuda.is_available():
    device = torch.device("cuda")  # объект устройства CUDA
    print('Using CUDA:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('Using CPU')

# Создать тензор и переместить его на GPU
x = torch.randn(300, 300, dtype=torch.float64).to(device)
t0 = time.time()

# Выполнить некоторые операции с тензором
for _ in tqdm.tqdm(range(int(1e+5))):
    y = torch.mm(x, x.transpose(0, 1))
    tt = torch.dot(x[0,:], x[1,:])


print(time.time()-t0)