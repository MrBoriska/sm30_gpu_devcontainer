FROM dizcza/pytorch-sm30:v1.10.2

USER root

# --- ИСПРАВЛЕНИЕ ОШИБКИ GPG KEY ---
# Удаляем старые репозитории NVIDIA, у которых истекли ключи.
# Они блокируют apt-get update, а для установки gcc они нам не нужны.
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ENV DEBIAN_FRONTEND=noninteractive

# Теперь устанавливаем компиляторы (нужны для сборки CuPy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreeimage3 \
    libfreeimage-dev \
    python3-tk tk-dev \
    && rm -rf /var/lib/apt/lists/*


# Обновляем pip (но не слишком сильно, так как Python 3.6 старый)
RUN python3 -m pip install --upgrade "pip<22.0" setuptools wheel

# --- НАСТРОЙКИ ДЛЯ CUPY (GeForce 660 Ti) ---

# 1. Указываем архитектуру Kepler (sm_30)
ENV CUPY_NVCC_GENERATE_CODE="arch=compute_30,code=sm_30"

# 2. Исправляем ошибки линковки (помогаем найти системные библиотеки)
ENV CFLAGS="-I/usr/include" 
ENV LDFLAGS="-L/usr/lib/x86_64-linux-gnu -ldl -lrt"

# 3. Устанавливаем CuPy версии 10.x
# Версия 10.6.0 — последняя стабильная ветка, идеально работающая с CUDA 10.2
# и Python 3.6. Более новые версии не соберутся.
RUN pip install "cupy==10.6.0"

# --- ДОПОЛНИТЕЛЬНЫЕ БИБЛИОТЕКИ (Опционально) ---
# Ставим старые версии библиотек, совместимые с Python 3.6
RUN pip install --no-cache-dir \
    "setuptools==59.5.0" \
    "numpy==1.22.4" \
    "pandas==1.4.4" \
    "matplotlib==3.5.3" \
    "scikit-learn==1.0.2"

WORKDIR /app

CMD ["/bin/bash"]
