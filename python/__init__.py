
import os

from PIL import Image
from video import preprocess_video

if __name__ == '__main__':

	frames : list[Image.Image] = preprocess_video('despacito 1337 - Trim.mp4', fill=(0, 255, 0))

	os.makedirs('frames', exist_ok=True)
	for index, frame in enumerate(frames):
		frame.save(f'frames/frame_{index}.jpg', optimize=True, quality=85)
