
from __future__ import annotations
from typing import Any
from PIL import Image, ImageFile

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

class VideoToDeltaFrames:

	filepath : str = None
	capture : VideoCapture = None
	stored_frames : list[Image.Image] = None

	def __init__( self ) -> VideoToDeltaFrames:
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

if __name__ == '__main__':

	translator = VideoToDeltaFrames()
	translator.load_video( 'chicken_trim.mp4' )

	frame : Image.Image = translator.next_frame()
	while frame != None:
		print( translator.get_current_index() - len(translator.stored_frames), '/', translator.get_total_length() )
		frame : Image.Image = translator.next_frame()
