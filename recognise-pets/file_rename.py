import os

# Функция для переименования файлов в директории с добавлением числового индекса и префикса
def rename_files_with_numbers(directory, prefix_name):    
    # Получаем список всех файлов в указанной директории
    files = os.listdir(directory)
    # Сортируем файлы по имени (алфавитный порядок)
    files.sort()

    # Проходим по каждому файлу и получаем его индекс и имя
    for index, filename in enumerate(files, start=1):    
        # Разделяем имя файла на имя и расширение
        file_name, file_extension = os.path.splitext(filename)
        
        # Формируем новое имя файла, добавляя префикс и числовой индекс
        new_name = f"{prefix_name}_{file_name}{file_extension}"
        
        # Получаем полный путь для старого и нового имени файла
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # Переименовываем файл
        os.rename(old_path, new_path)
        # Выводим сообщение о переименовании
        print(f"Файл переименован: {filename} -> {new_name}")

# Переименование файлов в папке с изображениями кошек
rename_files_with_numbers('./pet-images/Cat', 'cat')
# Переименование файлов в папке с изображениями собак
rename_files_with_numbers('./pet-images/Dog', 'dog')
