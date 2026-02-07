import torch
import torch.nn as nn

# import matplotlib
# matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt
import numpy as np

# 1. Проверка доступности GPU
if not torch.cuda.is_available():
    raise RuntimeError("GPU не найден! Проверьте драйвера или флаг --gpus all")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"--- ЗАПУСК НА {gpu_name} ---")
print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

# 2. Генерация данных (y = 2x + 1 + шум)
# Создаем тензоры сразу на GPU
N = 100
X = torch.linspace(-5, 5, N).view(-1, 1).to(device)
y_true = 2 * X + 1 + 0.5 * torch.randn(X.size()).to(device)

# 3. Создание простой модели (Линейная регрессия через нейросеть)
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
).to(device)

# 4. Настройка обучения
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("\nНачинаем обучение...")

# 5. Цикл обучения
epochs = 200
loss_history = []

for epoch in range(epochs):
    # Прямой проход
    y_pred = model(X)
    loss = criterion(y_pred, y_true)
    
    # Обратный проход
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("\nОбучение завершено!")

# 6. Проверка визуализации (если настроен X11)
try:
    # Переносим данные обратно на CPU для рисования
    X_cpu = X.cpu().numpy()
    y_true_cpu = y_true.cpu().numpy()
    y_pred_cpu = model(X).detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_cpu, y_true_cpu, label='Исходные данные')
    plt.plot(X_cpu, y_pred_cpu, color='red', label='Результат нейросети')
    plt.legend()
    plt.title(f"Результат на {gpu_name}")

    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title("График ошибки (Loss)")
    plt.xlabel("Эпохи")

    print("\nОткрываю окно с графиком...")
    plt.show()
    print("Тест успешно пройден.")

except Exception as e:
    print(f"\nОшибка при выводе графика: {e}")
    print("Но вычисления на GPU прошли успешно!")