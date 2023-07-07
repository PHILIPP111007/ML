import os
from PIL import Image


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

			# Открытие фотографии с помощью PIL
			image = Image.open(source_path)

			# Размеры исходной фотографии
			width, height = image.size

			# Размеры одной части (высота остается прежней)
			part_width = width // X
			part_height = height // Y

			# Перебор и сохранение каждой части фотографии
			for i in range(X):
				for j in range(Y):

					left = 0
					top = j * part_height
					right = part_width
					bottom = (j + 1) * part_height
					part_image = image.crop((left, top, right, bottom))

					# Генерация нового имени для части фотографии
					part_filename = f"{filename[:-4]}_{i + j + 1}.jpg"
					save_path = os.path.join(full_path, part_filename)

					# Сохранение части фотографии
					if not os.path.exists(save_path):
						part_image.save(save_path)

			# Закрытие исходного изображения
			image.close()


# Части фотографий
X, Y = 11, 11

if __name__ == '__main__':
	# Путь к папке с исходными фотографиями
	source_folder = input('Введите путь к исходному датасету:\n')

	# Путь к папке, куда нужно скопировать части фотографий
	dest_folder = input('Введите путь к новому датасету:\n')

	crop_save()
