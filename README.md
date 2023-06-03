# TSIG
Welcome to the Traffic Sign Image Generator 👋

This project implements a generation of artificial traffic sign images using **generative adversarial networks** (more specifically: **CycleGANs**).

<details open>
<summary>Examples</summary>

<div align=center>
<h2>UNet</h2>

![raw](https://user-images.githubusercontent.com/83597198/220124845-941839b9-3061-4f90-b1a4-943ab25b996c.gif) | ![motion_blur](https://user-images.githubusercontent.com/83597198/220123815-ad5a0949-1b44-4bf1-8921-e62346251feb.gif) | ![invalid](https://user-images.githubusercontent.com/83597198/220123835-f01edba3-22e4-4d49-8b73-36c1de62f0ff.gif) |
|:--:|:--:|:--:|
| **Raw** | **Motion Blur** | **Invalid Traffic Signs** |
</div>

<div align=center>
<h2>ResNet</h2> 

![resnet_raw](https://user-images.githubusercontent.com/83597198/220563790-d180cd2a-8e52-400b-8883-e1ccd00856c1.gif) | ![resnet_motion_blur](https://user-images.githubusercontent.com/83597198/220563976-6202825f-febd-4966-b534-9c542a21b46d.gif) | ![resnet_invalid](https://user-images.githubusercontent.com/83597198/220564027-1718936e-0521-460a-aba3-b88d605a3f12.gif) |
|:--:|:--:|:--:|
| **Raw** | **Motion Blur** | **Invalid Traffic Signs** |
</div>

</details>



# Getting Started
## Google Colab
<a href="https://colab.research.google.com/drive/1b8sW0nHd4J2G3D7BPG5zZ8DI1P8atOfP?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 
## Locally
### Troubleshoot
#### **[Windows]** Installation of Tensorflow Graphics fails because of OpenEXR
  ```bash
  $ pip install pipwin
  $ pipwin install openexr
  ```
  Then: re-run setup

# Documentation
<p>
  <a href="doc/Software%20Documentation/src" target="_blank">
    <img alt="software documentation" src="https://img.shields.io/badge/software%20documentation-html-blue" />
  </a>
  <a href="doc/dokumentation.pdf" target="_blank">
    <img alt="project documentation" src="https://img.shields.io/badge/project%20documentation-pdf-blue" />
  </a>
</p>

## Software Documentation
Live Server
```bash
$ pdoc --http localhost:8080 src
```

## About
### Dataset
TBD
