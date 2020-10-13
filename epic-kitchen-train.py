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
from random import shuffle
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

model = mtrn
for n, p in model.base_model.named_parameters():
    p.requires_grad = False

#Verb annotations: 0 - take, 1 - put, 9 - move
noun_annotations = 'EPIC_noun_classes.csv'
verb_annotations = 'EPIC_verb_classes.csv'

nouns = pd.read_csv(noun_annotations)
verbs = pd.read_csv(verb_annotations)

n1 = nouns['class_key'].to_numpy()
v1 = verbs['class_key'].to_numpy()


dataset_directory = 'dataset/'
x_ds = {'test': {'move': [], 'take': [], 'put': []}, 'train': {'move': [], 'take': [], 'put': []}}
y_ds = {'move': [9], 'take': [0], 'put': [1]}
xs = {'train': [], 'test': []}
ys = {'train': [], 'test': []}

pr = {'train': [], 'test': []}

for m in ['train', 'test']:
    for i in ['move', 'take', 'put']:
        current_directory = dataset_directory + m + '/' + i
        for f in os.listdir(current_directory):
            fn = current_directory + '/' + f
            
            frames = read_video(fn)
            frames = np.swapaxes(frames, 1, 3)
            
            pr[m].append((frames, y_ds[i]))

    shuffle(pr[m])
    for i in range(len(pr[m])):
        xs[m].append(pr[m][i][0])
        ys[m].append(pr[m][i][1])
    

batch_size = 1
segment_count = 8
snippet_length = 1
snippet_channels = 3
height, width = 224, 224

model = model.cuda()
s = 0
subs = []
print(frames.shape)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
total_loss = 0
total_correct = 0
for epoch in range(15):
    for m in ['train', 'test']:
        if m == 'train':
            model.train()
        else:
            model.eval()

        for j in range(len(xs[m])):
            label = torch.Tensor(ys[m][j]).long().cuda()
            inputs = torch.Tensor(xs[m][j])
            inputs = torch.reshape(inputs, (1, 24, height, width))
            inputs = inputs.cuda()
                

            with torch.set_grad_enabled(m == 'train'):
                verb_logits, noun_logits = model(inputs)
        
                _, pred = torch.max(verb_logits, 1)
                print('pred and label ', pred, label)
                loss = criterion(verb_logits, label)

                if m == 'train':
                    loss.backward()
                    optimizer.step()

            total_loss = total_loss + loss.item()
            total_correct += torch.sum(pred == label)
        

        print("epoch nb: ", epoch)
        print(m)
        print("total_loss: ", total_loss)
        print("avg_loss: ", total_loss/len(xs[m]))
        print("total_correct: ",total_correct)
        print("total_dataset: ", len(xs[m]))
        total_correct = 0


torch.save({'model_state-dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': total_loss, }, 'model.pth')

