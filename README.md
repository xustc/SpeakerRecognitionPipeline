# A Basic Speaker Recognition Pipeline

## dataset
In this pipeline I use THCHS-30 open-source dataset, you can get free access to it via 
http://openslr.org/18/

For higher accuracy, try http://openslr.org/33/ and http://openslr.org/38/ with slight change of data pre-processing.


## models
ResCNN and GRU model from Baidu(https://arxiv.org/abs/1705.02304) and 3D-CNN model by A.Torfi et al.(https://arxiv.org/abs/1705.09422) are used here.


## usage
### feature extraction
`python main.py extract`

### training
for softmax pre-training, use:

`python main.py train --triplet=False`

for fine-tunning based on triplets, use:

`python main.py train --triplet=True`

BTW, to understand triplets loss and its application, you can take a look at this video: https://www.youtube.com/watch?v=d2XB5-tuCWU

### test
So far only ResCNN and GRU models are supported.

`python main.py test`

### ONNX model exportation and deployment

first, run this

`python main.py export`

then, follow the instructions here(https://oracle.github.io/graphpipe/#/guide/servers/serving) to serve your model and do inference.

## TODO
- try larger dataset
- hyper-parameter tuning
- visualization
- rewrite some parts(e.g. Variable -> Tensor)
- fix bugs
- ...