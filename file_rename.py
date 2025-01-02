import os

def rename_files_with_numbers(directory, prefix_name):    
    files = os.listdir(directory)
    files.sort()

    for index, filename in enumerate(files, start=1):    
        file_name, file_extension = os.path.splitext(filename)
        
        # Формируем новое имя, добавляя число
        new_name = f"{prefix_name}_{file_name}{file_extension}"
        
        # Получаем полный путь для старого и нового имени файла
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # Переименовываем файл
        os.rename(old_path, new_path)
        print(f"Файл переименован: {filename} -> {new_name}")


rename_files_with_numbers('./pet-images/Cat', 'cat')
rename_files_with_numbers('./pet-images/Dog', 'dog')
