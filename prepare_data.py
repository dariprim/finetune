import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Загружаем CSV
print("Загрузка данных...")
df = pd.read_csv('psiholog_2023_12_16.csv')

# Проверяем, что нужные колонки существуют
required_columns = ['question_body', 'answers']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Колонка {col} не найдена в CSV. Доступные колонки: {df.columns.tolist()}")

# Удаляем пустые значения
df = df.dropna(subset=['question_body', 'answers'])

# Очищаем текст от лишних пробелов
df['question_body'] = df['question_body'].str.strip()
df['answers'] = df['answers'].str.strip()

# Удаляем слишком короткие вопросы или ответы (менее 10 символов)
df = df[df['question_body'].str.len() >= 10]
df = df[df['answers'].str.len() >= 10]

print(f"Всего примеров после очистки: {len(df)}")

# Форматируем данные для обучения в диалоговом формате YandexGPT
def format_conversation(row):
    """
    Преобразует строку в формат, ожидаемый моделью YandexGPT:
    {вопрос пользователя} Ассистент:[SEP] {ответ психолога}</s>
    """
    user_text = row['question_body'].replace('\n', ' [NL] ')
    assistant_text = row['answers'].replace('\n', ' [NL] ')
    
    return {
        "text": f"{user_text} Ассистент:[SEP] {assistant_text}</s>"
    }

# Применяем форматирование
formatted_data = []
for _, row in df.iterrows():
    formatted_data.append(format_conversation(row))

# Разделяем на обучающую и валидационную выборки (90% / 10%)
train_data, val_data = train_test_split(formatted_data, test_size=0.1, random_state=42)

print(f"Обучающих примеров: {len(train_data)}")
print(f"Валидационных примеров: {len(val_data)}")

# Сохраняем в JSONL формат (по одной строке JSON на пример)
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("Данные сохранены в train.jsonl и val.jsonl")

# Покажем пример
print("\nПример форматированных данных:")
print(train_data[0]['text'][:200] + "...")