import os
from pydub import AudioSegment

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for genre in genres:
    aud_path = os.path.join('content/audio3sec', f'{genre}')
    os.makedirs(aud_path)

i = 0
for genre in genres:
    j = 0
    for filename in os.listdir(f'genres/{genre}'):
        song = f'genres/{genre}/{filename}'

        j = j + 1
        for w in range(0, 10):
            i = i + 1
            t1 = 3 * (w) * 1000
            t2 = 3 * (w + 1) * 1000
            newAudio = AudioSegment.from_file(song, 'au')
            new = newAudio[t1:t2]
            new.export(f'content/audio3sec/{genre}/{genre + str(j) + str(w)}.wav', format="wav")