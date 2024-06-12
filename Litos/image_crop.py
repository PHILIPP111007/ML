import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop_save():

	# Создание папки назначения, если она не существует
	if not os.path.exists(dest_folder):
		os.makedirs(dest_folder)

	# Перебор всех файлов в папке с исходными фотографиями
	for filename in os.listdir(source_folder):
		if filename.endswith(".jpg"):

			# Получение полного пути к исходному файлу
			source_path = os.path.join(source_folder, filename)

			# Новый путь
			full_path = os.path.join(dest_folder, filename[:-4])

			if not os.path.exists(full_path):
				os.makedirs(full_path)

			image = Image.open(source_path)
			width = image.size[0] // W
			height = image.size[1] // H

			# Перебор и сохранение каждой части фотографии
			for i in range(H):
				for j in range(W):

					# Обрезаем изображение
					part_image = image.crop(
						(
							j * width, # left
							i * height, # top
							j * width + width, # right
							i * height + height # bottom
						)
					)

					# Генерация нового имени для части фотографии
					part_filename = f"{filename[:-4]}_{i + j + 1}.jpg"
					# Путь для сохранения фотографии
					save_path = os.path.join(full_path, part_filename)

					# Сохранение
					if not os.path.exists(save_path):
						part_image.save(save_path)
					
					# part_image.show()

			image.close()


# Разбиение на части фотографии, можно настраивать
W, H = 2, 2

if __name__ == '__main__':
	# Путь к папке с исходными фотографиями
	source_folder = input('Введите путь к исходному датасету:\n')

	# Путь к папке, куда нужно скопировать части фотографий
	dest_folder = input('Введите путь к новому датасету:\n')

	crop_save()
