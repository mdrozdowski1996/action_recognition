import pandas as pd
import torch.hub
import torchvision
import moviepy
import cv2
import numpy as np
from torchvision.datasets.video_utils import VideoClips
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import editor
import os


def annotate(clip, txt, txt_color='red', fontsize=50, font='Xolonium-Bold'):
    txtclip = editor.TextClip(txt, fontsize=fontsize, font=font, color=txt_color)
    cvc = editor.CompositeVideoClip([clip, txtclip.set_pos(('center', 'bottom'))])
    return cvc.set_duration(clip.duration)

def read_video(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    ret = True
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 5
            cap.set(1, count)
            img = frame
            img = cv2.resize(frame, (224, 224))
            a = np.asarray(img)
            frames.append(a)
        else:
            cap.release()
            break

    ret_frames = []
    l = int((len(frames)-1)/8)

    for i in range(8):
        ret_frames.append(frames[i*l])

    cap.release()
    return np.asarray(ret_frames)


repo = 'epic-kitchens/action-models'

class_counts = (125, 352)
segment_count = 8
base_model = 'resnet50'
tsn = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                     base_model=base_model, 
                     pretrained='epic-kitchens', force_reload=True)
trn = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                     base_model=base_model, 
                     pretrained='epic-kitchens')
mtrn = torch.hub.load(repo, 'MTRN', class_counts, segment_count, 'RGB',
                     base_model=base_model, 
                     pretrained='epic-kitchens')

#Verb annotations: 0 - take, 1 - put, 9 - move
noun_annotations = 'EPIC_noun_classes.csv'
verb_annotations = 'EPIC_verb_classes.csv'

nouns = pd.read_csv(noun_annotations)
verbs = pd.read_csv(verb_annotations)

n1 = nouns['class_key'].to_numpy()
v1 = verbs['class_key'].to_numpy()


test_directory = 'dataset/test/'
test_ds = {'move': [], 'take': [], 'put': []}

y_ds = {'move': 9, 'take': 0, 'put': 1}

for i in ['move', 'take', 'put']:
    for f in os.listdir(test_directory + i):
        fn = test_directory + i + '/' + f
        
        frames = read_video(fn)
        frames = np.swapaxes(frames, 1, 3)
        test_ds[i].append(frames)


height, width = 224, 224

model = trn
model = model.cuda()
s = 0
subs = []

criterion = torch.nn.CrossEntropyLoss()
total_loss = 0
dataset_size = 0
total_correct = 0

for i in ['move', 'put', 'take']:

    label = torch.Tensor([y_ds[i]]).long().cuda()

    
    for f in test_ds[i]:

        inputs = torch.Tensor(f)
        inputs = torch.reshape(inputs, (1, 24, height, width))
        inputs = inputs.cuda()
        
        features = model.features(inputs)
        verb_logits, noun_logits = model.logits(features)
        sm = torch.nn.Softmax(dim = 1)
        verb_sm = sm(verb_logits)
        verb_nb = torch.topk(verb_sm, 1)

        verb_val = verb_nb.values.cpu().detach().numpy()[0][0]
        verb_class = verb_nb.indices.cpu().detach().numpy()[0][0]

        print('class ', y_ds[i], verb_class) 
        if y_ds[i] == verb_class and verb_val > 0.5:
            total_correct += 1

        loss = criterion(verb_sm, label)
        total_loss = loss.item()
        
        dataset_size += 1
        print(i, " ",  verb_sm[0][0], verb_sm[0][1], verb_sm[0][9], verb_nb)



print(total_loss, dataset_size, total_correct, total_correct/dataset_size, total_loss/dataset_size)

