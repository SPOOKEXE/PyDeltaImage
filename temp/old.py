
import numpy as np
import cv2
import os

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRANSITION_IGNORE_COLOR = (0, 0, 0)
TRANSITION_PARSE_COLOR = (255, 255, 255)

def find_image_delta( current : list, next : list ) -> list:
	items = (np.array(next) - np.array(current)).tolist()
	callback = lambda rgb : (rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0) and TRANSITION_IGNORE_COLOR or rgb
	new_items = []
	for row in items:
		new_items.append([ callback(tuple(rgb)) for rgb in row ])
	return new_items

def get_frames( filepath : str ) -> list[Image.Image]:
	vidObj = cv2.VideoCapture( filepath )
	frames : list = []
	count : int = 0
	while True:
		success, image = vidObj.read()
		if success == False: break
		pil = Image.fromarray( cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )
		frames.append( pil )
		count += 1
	return frames

def find_different_pixels( filepath : str ) -> list[Image.Image]:
	frames : list[Image.Image] = []
	is_first : bool = True
	raw_frames = get_frames( filepath )
	os.makedirs('raw', exist_ok=True)
	os.makedirs('delta', exist_ok=True)
	for index, item in enumerate(raw_frames):
		item.save(f'raw/frame_{index}.jpg')
		if is_first == True:
			frames.append( item )
			is_first = False
			continue
		delta0 = find_image_delta( np.array(raw_frames[index-1]).tolist(), np.array(raw_frames[index]).tolist() )
		delta1 = Image.fromarray( np.array(delta0), mode='RGB' )
		delta1.save(f'delta/frame_{index-1}.jpg')
	return frames
frames : list[Image.Image] = find_different_pixels( 'chicken_trim.mp4' )
