import os

def rename_files_with_numbers(directory):    
    files = os.listdir(directory)
    files.sort()

    for index, filename in enumerate(files, start=1):    
        file_name, file_extension = os.path.splitext(filename)
        file_name = file_name[8:]
        # Формируем новое имя, добавляя число в начало
        new_name = f"dog_{file_name}{file_extension}"
        
        # Получаем полный путь для старого и нового имени файла
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # Переименовываем файл
        os.rename(old_path, new_path)
        print(f"Файл переименован: {filename} -> {new_name}")


rename_files_with_numbers('./dogs-vs-cats/1')
