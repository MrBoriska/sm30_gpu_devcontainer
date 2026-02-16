from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import time
import warnings
from transformers import logging as transformers_logging

# Подавляем конкретные предупреждения
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
transformers_logging.set_verbosity_error()  # убирает сообщения о специальных токенах и прочие info/warning от transformers

# Название модели
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)
model.to("cuda")

# Устанавливаем pad_token_id, если не задан
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Системное сообщение с чёткими инструкциями и примерами
system_message = (
    "Ты — робот-помощник. Твоя задача — по запросу пользователя вернуть одну из трёх команд на английском языке:\n"
    "- forward (движение вперёд)\n"
    "- turn left (поворот налево)\n"
    "- turn right (поворот направо)\n\n"
    "Отвечай только одним словом (командой) без каких-либо пояснений, знаков препинания или дополнительного текста.\n\n"
    "Примеры:\n"
    "Пользователь: Двигайся вперёд\n"
    "Ты: forward\n\n"
    "Пользователь: Поверни налево\n"
    "Ты: turn left\n\n"
    "Пользователь: Поверни направо\n"
    "Ты: turn right"
)

# Запрос пользователя (можно менять)
user_query = "follow me"

# Формируем диалог в формате, который ожидает модель
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_query}
]

# Применяем шаблон чата для получения входного текста
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # добавляет токен начала ответа ассистента
)

# Токенизируем
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Стример для потокового вывода
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,           # не выводим промпт
    skip_special_tokens=True    # не выводим специальные токены
)

# Засекаем время
start_time = time.time()

# Генерация (жадный поиск, без случайности)
outputs = model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=10,           # команда короткая
    pad_token_id=tokenizer.pad_token_id,
    do_sample=False,             # детерминированный выбор
    num_beams=1,                  # жадный поиск
    temperature=None,             # убираем параметры для sample, чтобы избежать предупреждений
    top_p=None,
    top_k=None
)

# Время окончания
end_time = time.time()
generation_time = end_time - start_time

# Подсчёт сгенерированных токенов
full_output_ids = outputs[0]
prompt_length = inputs['input_ids'].shape[1]   # длина промпта (с chat template)
new_tokens = len(full_output_ids) - prompt_length

print(f"\n\n--- Статистика ---")
print(f"Время генерации: {generation_time:.2f} сек")
print(f"Сгенерировано токенов: {new_tokens}")
print(f"Скорость: {new_tokens / generation_time:.2f} токенов/сек")