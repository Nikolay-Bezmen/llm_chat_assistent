# Проект обучения LLM: Практическая реализация

## Описание проекта
Проект по обучению языковой модели (LLM) с нуля с использованием архитектуры GPT. Включает:
- Подготовку данных из открытых источников
- Реализацию модели с использованием PyTorch
- Процесс обучения с отслеживанием метрик
- Генерацию текста на основе обученной модели

## Основные компоненты
- **Модель**: 
  ```python
  GPT(
    vocab_size=50257,
    n_embd=512,
    n_head=8,
    block_size=256,
    n_layer=16,
    dropout=0.3
  )
**Оптимизация и настройки**:
- **Оптимизатор**: AdamW с параметрами:
  ```python
  learning_rate = 1e-2
  weight_decay = 0.01
  grad_clip = 1.0  # Клиппирование градиентов
  использовалось уменьшение lr с 1e-2 до 0 при помощи CosineAnnealingLR
**Достигнутые результаты**
- начальный Loss : 10.9
- достигнутый Loss : 4.2
- начальное Perplexity : 58000+-
- достигнутое Perplexity : 60+-
  
