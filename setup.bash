#!/usr/bin/env bash

conda create -n cyclegan
conda activate cyclegan

pip install -r requirements.txt

curl -o 'src/external_tools' 'https://github.com/LynnHo/CycleGAN-Tensorflow-2/blob/master/module.py'
mv src/external_tools/module.py src/external_tools/resnet.py