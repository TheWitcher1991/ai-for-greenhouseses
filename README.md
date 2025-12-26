# AI определения признаков растений, прогнозирования развития болезней и вредителей

## Быстрый старт (локально)

Клонируйте репозиторий:
```bash
git clone https://github.com/TheWitcher1991/ai-for-greenhouseses.git
cd ai-for-greenhouseses
```

Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate        # Windows
poetry install
```

## Обучение моделей
- Скрипт/команда для обучения: `python ai/train.py`
- Общие шаги:
  1. Подготовить датасет (директория `data`).
  2. Запустить обучение: `python ai/train.py`

## Инференс / Предсказание
- Для реального времени: сервис подписывается на MQTT и возвращает рекомендации/команды.
- Пример запроса к HTTP API:
  ```http
  POST /api/mlm/predict
  Content-Type: application/json

  {
    "file": UploadFile,
  }
  ```
- Скрипт для оффлайн-инференса:
  ```bash
  python ai/predict.py
  ```
