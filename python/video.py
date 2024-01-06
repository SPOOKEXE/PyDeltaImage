
from __future__ import annotations
from typing import Any
from PIL import Image, ImageFile

import matplotlib.pyplot as plt
import traceback
import numpy as np
import cv2
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

class VideoCapture:
	# file data
	filepath : str = None
	capture : cv2.VideoCapture = None
	total_length : int = -1
	width : int = -1
	height : int = -1
	fps : float = -1.0
	# frame data
	current_index : int = 0
	current_frame : Any = None

	def __init__( self, filepath : str ) -> VideoCapture:
		self.filepath = filepath
		self.restart()
		self._update_props()

	def _update_props( self ) -> None:
		self.total_length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
		self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

	def next_frame( self ) -> bool:
		success, frame = self.capture.read()
		if success == False:
			self.current_frame = None
			# self.current_index = -1
			return False
		self.current_frame = Image.fromarray( cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
		self.current_index += 1
		return True

	def restart( self ) -> None:
		self.capture = cv2.VideoCapture( self.filepath )
		self.current_index = 0
		self.current_frame = None

class VideoDecoder:

	filepath : str = None
	capture : VideoCapture = None
	stored_frames : list[Image.Image] = None

	def __init__( self ) -> VideoDecoder:
		pass

	def release_capture( self ) -> None:
		'''Release the capture.'''
		self.filepath = None
		self.capture = None

	def load_video( self, filepath : str ) -> bool:
		'''Load a video.'''
		try:
			self.filepath = filepath
			self.capture = VideoCapture( filepath )
			self.stored_frames = list()
			return True
		except Exception as exception:
			print('Failed to load video file at filepath: ', filepath)
			traceback.print_exception( exception )
			self.release_capture( )
			return False

	def is_video_loaded( self ) -> bool:
		'''Is the video loaded?'''
		return self.capture != None

	def is_finished( self ) -> bool:
		'''Is the video finished?'''
		return self.is_video_loaded() == False or self.capture.current_index >= self.capture.total_length

	def preprocess_next( self ) -> bool:
		'''Process the next frame - useful for low memory.'''
		if self.is_video_loaded( ) == False:
			return False
		if self.capture.next_frame( ) == False:
			return False
		self.stored_frames.append( self.capture.current_frame )
		return True

	def preprocess_entire( self ) -> bool:
		'''Preprocess the entire video - takes up heaps of memory.'''
		if self.is_video_loaded( ) == False:
			return False
		while self.preprocess_next( ) == True:
			pass
		return True

	def get_total_length( self ) -> int | None:
		'''Get the total length of the video at the filepath.'''
		if self.is_video_loaded( ) == False:
			return None
		return self.capture.total_length

	def get_current_index( self ) -> int | None:
		'''Get the current frame'''
		if self.is_video_loaded( ) == False:
			return None
		return self.capture.current_index

	def get_stored_frames_count( self ) -> int | None:
		'''Get the total available frames that have been preprocessed'''
		if self.is_video_loaded( ) == False:
			return None
		return len(self.stored_frames)

	def restart( self ) -> None:
		self.capture.restart()

	def next_frame( self ) -> Image.Image | None:
		'''
		Get the next frame - will automatically process_next if no frames are available.
		Run 'preprocess_entire' if you have enough memory for the video for fast performance.
		'''
		if self.is_video_loaded( ) == False:
			return None
		if len( self.stored_frames ) != 0:
			return self.stored_frames.pop(0)
		if self.preprocess_next( ) == False:
			return None
		return self.stored_frames.pop(0)

class ImageEditor:

	@staticmethod
	def to_cv2( img : Image.Image ) -> np.ndarray:
		return cv2.cvtColor( np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR )

	@staticmethod
	def to_PIL( img : np.ndarray, mode : str = 'RGB' ) -> Image.Image:
		return Image.fromarray( cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mode=mode )

	@staticmethod
	def get_diff_mask( image1 : Image.Image, image2 : Image.Image, threshold : float = 1.0 ) -> Image.Image:
		'''Return an image that is the difference between these two images. White pixels are the difference.'''
		img1 = ImageEditor.to_cv2( image1 )
		img2 = ImageEditor.to_cv2( image2 )
		diff = cv2.absdiff(img1, img2)
		mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
		imask = (mask > threshold)
		canvas = np.zeros_like(img2, np.uint8)
		canvas[imask] = 255
		return ImageEditor.to_PIL(canvas, mode='RGB')

	@staticmethod
	def get_diff_color( image1 : Image.Image, image2 : Image.Image, fill : tuple = (0, 255, 0), threshold : float = 1.0 ) -> Image.Image:
		img1 = ImageEditor.to_cv2( image1 )
		img2 = ImageEditor.to_cv2( image2 )
		diff = cv2.absdiff(img1, img2)
		imask = (diff > threshold)
		canvas = np.full_like(img2, fill, img1.dtype)
		canvas[imask] = img2[imask]
		return ImageEditor.to_PIL(canvas, mode='RGB')

	@staticmethod
	def to_gaussian_blur( image : Image.Image, blur : int = 3, deviation : int = 0 ) -> Image.Image:
		'''Convert the (assumed difference) image to a heatmap image'''
		blurred = cv2.GaussianBlur( ImageEditor.to_cv2(image.convert('RGB')), (blur, blur), deviation)
		return ImageEditor.to_PIL( blurred )

def preprocess_video( filepath : str, fill : tuple = (0, 255, 0) ) -> list[Image.Image]:

	decoder = VideoDecoder()
	decoder.load_video( filepath )

	# deltavideo = cv2.VideoWriter('deltaoutput.mp4', 0, decoder.capture.fps, (decoder.capture.width, decoder.capture.height))
	# video = cv2.VideoWriter('output.mp4', 0, decoder.capture.fps, (decoder.capture.width, decoder.capture.height))

	frames : list[Image.Image] = []
	previous : Image.Image = None
	while True:
		frame : Image.Image = decoder.next_frame()
		if frame == None:
			break

		index : int = decoder.get_current_index() - len(decoder.stored_frames)
		if previous != None:
			diff = ImageEditor.get_diff_color( previous, frame, fill=fill, threshold=1 )
			# diff.save(f'image/diff_{index}.jpg')
			frames.append( diff )
			# save_frame = ImageEditor.to_cv2( diff )
			# imask = ( save_frame != (0, 255, 0) )
			# canvas = ImageEditor.to_cv2(previous)
			# canvas[imask] = save_frame[imask]
			# deltavideo.write( ImageEditor.to_cv2(diff) )
			# video.write( canvas )
		previous = frame
		print( index, '/', decoder.get_total_length() )

	# deltavideo.release()
	# video.release()
	decoder.release_capture()
	return frames
