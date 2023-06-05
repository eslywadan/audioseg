import os
from pydub import AudioSegment

src = os.path.join('dataset/music/', 'FMP_C8_F27_Chopin_Op028-04_minor.mp3')
dst =  os.path.join('dataset/music/', 'FMP_C8_F27_Chopin_Op028-04_minor.wav')

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")