import torch.hub
import time
import csv
import numpy as np

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

for entrypoint in torch.hub.list(repo):
    print(entrypoint)
    print(torch.hub.help(repo, entrypoint))

batch_size = 1
segment_count = 8
snippet_length = 1
snippet_channels = 3
height, width = 224, 224
res = {}

for batch_size in [1, 2, 4]:
    print("batch_size: ", batch_size)
    inputs1_ = torch.randn(
        [batch_size, segment_count, snippet_length, snippet_channels, height, width]
    )
    
    inputs1_ = inputs1_.reshape((batch_size, -1, height, width))
    inputs1 = inputs1_.cuda()

    batch_results = {}
    for (model, model_name) in [(tsn, "TSN"), (trn, "TRN"), (mtrn, "MTRN")]:
        print(model_name)
        t_f = []
        t_l = []
        t_t = []
        model_results = {}
        model.cuda()
        for i in range(30):

            start = time.time()
            verb_logits, noun_logits = model(inputs1)
            end = time.time()

            t_t.append(end - start)
            print("time : ", end - start)

        a_t = np.mean(t_t[5:])

        model_results['latency'] = a_t
        model_results['throughput'] = batch_size / a_t

        batch_results[model_name] = model_results

        print("avg time total: ", a_t)

    res[batch_size] = batch_results

print(res)
