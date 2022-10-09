<p align="center">
  <a href="https://example.com/">
    <img src="https://via.placeholder.com/72" alt="Logo" width=72 height=72>
  </a>

  <h3 align="center">Art Style Challenge</h3>

  <p align="center">
    194.077 – Applied Deep Learning
  </p>
</p>


## Table of contents

- [Idea Description](#idea-description)
- [State of the Art](#state-of-the-art)
- [Methodology](#methodology)
- [Timeline](#timeline)
- [References](#references)

## Idea Description

**Topic:**
Creating art requires skill and tools and can be time-consuming.
Coming from a graphic design background, I was eager to use deep learning approaches as a substitute for conventional illustration methods. 
In previous work, I already experimented with StyleGANs to generate characters from the *Animal Crossing* franchise. 
For this project, I aim to take part in the famous online drawing challenge called: *Art Style Challenge*.
It was invented around 6 years ago by a 17-year-old Instagrammer called *beautifulness87*. 
The goal was to redraw the same character in as many art styles as possible, including famous shows like *Adventure Time*, *Steven Universe*, *Gravity Falls*, or even studio styles like *Disney* or *Ghibli*. 
The options are almost limitless.
The header of this README shows my own take on the Art Style Challenge from 4 years ago.

However, instead of drawing them by hand, which requires multiple hours and lots of skill, we utilize StyleGANs.
Gathering our own data we either train our model using transfer learning or train a new conditional GAN from scratch. 
We further apply style transfer by mixing the weights of StyleGANs within different domains [[2]](#2).
Our goal is to generate deceptively real-looking characters in a minimum of about 5 different styles. 

**Type:**
Bring your own data 

## State of the Art

The use of StyleGANs is a currently explored research topic and can be deployed in a variety of applications in the field of image generation. 
For our study, we further explore style transfer utilizing StyleGANs.
State-of-the-art approaches show promising results for gaining more control over a style on different levels of detail. 


- **Higher control over the outcome:** 
Pinkney et al. [[2]](#2) proposed a method for controllable image synthesis by fine-tuning the interpolation of different resolution layers in the model. 
Each layer responds to different features.
While the lower levels address the shape and geometry of the face, higher resolution layers are responsible for the texture, color, and lighting. 
Based on StyleGAN2 the method starts with a pre-trained model and uses transfer learning for the new dataset. 
Instead of linearly interpolating between all the parameters, an arbitrary function is used to mix the weights from both the original and new generators.
The examples chose a binary decision between either turning a new layer on or off, called ``layer swapping''. 
Swapping the layer can for instance enable the preservation of the face shape and only apply texture and color of the new style, enabling more control over the outcome.
- **Less data:**
Training a generative adversarial network often requires a vast amount of data.
Therefore, finding and retrieving the images of a specific art style poses one of the main challenges. 
Available samples of a certain quality are limited and time-consuming to collect. 
However, simply using too little data often leads to overfitting and unsatisfactory results. 
Using the standard solution of data augmentation might lead to unwanted artifacts, especially when trying to copy a style. 
Karras et al. [[5]](#5) introduces a variation called adaptive discriminator augmentation (ADA) which significantly stabilizes training with limited data.
Additional test show, that transfer learning also benefits from ADA which further reduces the training data requirements. 
- **Closer resemblance:** 
For the last improvement, we aim for a higher resemblance between the stylized image and the original person. 
Song et al. [[3]](#3) achieves good generalization of different portrait styles using inversion-consistent transfer learning.
Furthermore, it efficiently encodes different levels of details by augmenting a multi-resolution latent space.
To better match the original appearance, an attribute-aware generator is introduced. Artists often apply different stylizations depending on their gender, age, and other attributes. 
AgileGANs multi-path generator takes this theory into account and further enhances features based on different attributes. 

## Methodology

1. **State-of-the-art:**
First, we set up the state-of-the-art papers and their corresponding repositories as listed in the following Table.

|                             Repository                            |                  Description                 |               Paper              |
|:-----------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------:|
|   <a href="https://github.com/NVlabs/stylegan2-ada-pytorch">StyleGAN2</a> | official pytorch implementation of StyleGAN2 | [[4]](#4) |
|       <a href="https://github.com/NVlabs/stylegan3">StyleGAN3</a>      |         official pytorch implementation of StyleGAN3         |  [[1]](#1)  |

We aim to compare the results of our Style Challenge generated by StyleGAN2 as well as StyleGAN3.

2. **Dataset:**
Search for suitable datasets of images and faces that define domains similar to the FFHQ dataset.
We aim to semi-automize our collecting process, by introducing image scrappers or applying face recognition on videos and extracting the faces.
Last we preprocess our data, by aligning and cropping the images and use the tools provided by the repository to generate a matching database for our models.

3. **Training:**
We train multiple models utilizing transfer learning and evaluate their progress using the Frichet Inception Distance (FID) during the training process.
Additionally, we train a conditional StyleGAN including all selected art styles. 

4. **Build Application:** 
We build a small web application. 
Depending on the remaining time, it will either show all the trained styles on randomly generated people or the user can upload an image of a face and the styles will be applied.

## Timeline

|      Milestone   |  Time Estimation |  Deadline  |
|:----------------:|:----------------:|-----------:|
|   Project Set-Up |        4h        | 14.12.2022 |
|  Data Collection |        10h       | 14.12.2022 |
|     Prototype    |        12h       | 14.12.2022 |
| Final Implementation |    18h       | 18.01.2023 |
|      Report      |        10h       | 18.01.2023 |
| Final Presentation |      4h        | 26.01.2023 |

## References

<a id="1">[1]</a> 
Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko
Lehtinen, and Timo Aila. Alias-free generative adversarial networks. *Advances in
Neural Information Processing Systems*, 34, 2021.

<a id="2">[2]</a> 
Justin NM Pinkney and Doron Adler. Resolution dependent gan interpolation for
controllable image synthesis between domains. *Machine Learning for Creativity and
Design NeurIPS 2020 Workshop*, 2020.

<a id="3">[3]</a> 
Guoxian Song, Linjie Luo, Jing Liu, Wan-Chun Ma, Chunpong Lai, Chuanxia Zheng,
and Tat-Jen Cham. Agilegan: stylizing portraits by inversion-consistent transfer
learning. *ACM Transactions on Graphics (TOG)*, 40(4):1–13, 2021.

<a id="4">[4]</a> 
Yuri Viazovetskyi, Vladimir Ivashkin, and Evgeny Kashin. Stylegan2 distillation
for feed-forward image manipulation. *In European Conference on Computer Vision,
pages 170–186. Springer*, 2020.

<a id="5">[5]</a> 
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila.
Training generative adversarial networks with limited data. Advances in Neural Information
Processing Systems, 33:12104–12114, 2020.
