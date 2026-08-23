"""
download_dataset_workaround.py

Обходной путь для известной проблемы на Windows: встроенная в ogbench
функция download_datasets() не закрывает файл перед os.rename(), из-за
чего Windows (в отличие от macOS/Linux) выдаёт ошибку "файл занят другим
процессом". Здесь тот же датасет скачивается вручную, но с явным
закрытием файла — ogbench увидит, что файлы уже на месте, и просто
пропустит собственную загрузку.

Запуск (один раз, после активации venv):
    python download_dataset_workaround.py
"""
import os
import urllib.request

DATASET_DIR = os.path.expanduser('~/.ogbench/data')
DATASET_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'

DATASET_NAMES = ['antmaze-medium-navigate-v0']

os.makedirs(DATASET_DIR, exist_ok=True)

file_names = []
for name in DATASET_NAMES:
    file_names.append(f'{name}.npz')
    file_names.append(f'{name}-val.npz')

for file_name in file_names:
    file_path = os.path.join(DATASET_DIR, file_name)
    if os.path.exists(file_path):
        print(f'Уже скачано, пропускаю: {file_name}')
        continue

    url = f'{DATASET_URL}/{file_name}'
    print(f'Скачиваю: {url}')

    response = urllib.request.urlopen(url)
    total_size = getattr(response, 'length', None)
    downloaded = 0

    # Пишем СРАЗУ в финальное имя (без .tmp и без os.rename) —
    # это и есть обход бага. Файл закрывается автоматически через
    # "with", ДО того как мы вообще пытаемся что-то переименовывать.
    with open(file_path, 'wb') as f:
        while True:
            chunk = response.read(1024 * 1024)  # по 1 МБ за раз
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded / total_size * 100
                print(f'\r  {downloaded / 1e6:.1f} МБ / {total_size / 1e6:.1f} МБ ({percent:.0f}%)', end='')
        print()

    print(f'Готово: {file_name}')

print('\nВсе файлы датасета на месте. Теперь можно запускать run_compare_eval.py')
