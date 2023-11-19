from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import subprocess
import json
import os
from tran_rewt import prepinaniy
from recasepunc.recasepunc import WordpieceTokenizer
from recasepunc.recasepunc import Config
SetLogLevel(0)

# Проверяем наличие модели
if not os.path.exists("model"):
    print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit (1)

# Устанавливаем Frame Rate
FRAME_RATE = 16000
CHANNELS=1

model = Model("vosk-model-ru-0.10")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)

# Используя библиотеку pydub делаем предобработку аудио
mp3 = AudioSegment.from_mp3('респ-8-1-_2_.mp3')
mp3 = mp3.set_channels(CHANNELS)
mp3 = mp3.set_frame_rate(FRAME_RATE)

# Преобразуем вывод в json
rec.AcceptWaveform(mp3.raw_data)
result = rec.Result()
text = json.loads(result)["text"]
# print(text)
# text = 'должен с ним отлично юля давайте теперь с вами тоже познакомимся расскажите немножечко о себе где вы работаете где живете какие у вас есть хобби а звук воронеже двадцать девять лет работаю в сфере'

# Добавляем пунктуацию
cased = prepinaniy(text)#subprocess.check_output('python recasepunc/recasepunc.py predict recasepunc/checkpoint', stderr=subprocess.STDOUT, shell=True, text=True, input=text)

# Записываем результат в файл "data.txt"
with open('data.txt', 'w') as f:
    json.dump(cased, f, ensure_ascii=False, indent=4)