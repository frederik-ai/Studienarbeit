# TSIG
Welcome to the Traffic Sign Image Generator 👋

This project implements a generation of artificial traffic sign images using **generative adversarial networks** (more specifically: **CycleGANs**).

<details open>
<summary>Examples</summary>

![raw](https://user-images.githubusercontent.com/83597198/220124845-941839b9-3061-4f90-b1a4-943ab25b996c.gif) | ![motion_blur](https://user-images.githubusercontent.com/83597198/220123815-ad5a0949-1b44-4bf1-8921-e62346251feb.gif) | ![invalid](https://user-images.githubusercontent.com/83597198/220123835-f01edba3-22e4-4d49-8b73-36c1de62f0ff.gif) |
|:--:|:--:|:--:|
| **Raw** | **Motion Blur** | **Invalid Traffic Signs** |

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
- German Traffic Sign Recognition Benchmark
<br> https://benchmark.ini.rub.de/gtsrb_dataset.html (20.10.2022)
- 43 classes, 50,000 images
