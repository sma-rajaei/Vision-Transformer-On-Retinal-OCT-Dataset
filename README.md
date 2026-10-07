# Retinal OCT Image Classification using Vision Transformers architecture built from scratch


## Overview

This project presents a complete **Vision Transformer (ViT)** architecture built entirely from scratch for Retinal OCT image classification. Instead of relying on pre-built implementations, all core components, including Patch Embedding, CLS Token, Positional Embedding, Multi-Head Self-Attention, MLP Blocks, and Transformer Encoder Blocks, are manually implemented.
The model is trained on a multi-class Retinal OCT dataset to explore the ability of Transformer-based architectures in learning global relationships within medical images and performing accurate disease classification.
This project combines Deep Learning, Computer Vision, and Medical Image Analysis with a focus on understanding and implementing the internal mechanisms of Vision Transformers.

## Research Motivation

Retinal OCT provides detailed cross-sectional images of retinal structures and plays an important role in medical diagnosis. Accurate analysis of these images can help identify and differentiate various retinal conditions.

While CNNs have shown strong performance in medical image analysis, they mainly learn local features through convolution operations, making the understanding of long-range relationships between distant image regions more challenging.

Vision Transformers overcome this limitation by enabling direct interaction between all image patches through the self-attention mechanism. This allows the model to capture global dependencies and complex patterns from the early stages of feature learning.

The goal of this project is to build a Vision Transformer from the ground up and investigate its capability in medical image classification, combining theoretical understanding with practical implementation in Computer Vision.

# Dataset

<img width="790" height="212" alt="image" src="https://github.com/user-attachments/assets/badabf0f-4267-406c-8334-64e5d1bd22e6" />


The project uses a retinal Optical Coherence Tomography (OCT) image dataset for multi-class classification.
link to the Dataset in Kaggle : https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8

The number of samples in each class is balanced:

<img width="560" height="413" alt="image-1" src="https://github.com/user-attachments/assets/1c2ec614-3e14-433a-b021-f2a47fbc7e59" />


The images have 7 different sizes, and their distribution is:

<img width="695" height="497" alt="image-2" src="https://github.com/user-attachments/assets/cc197db2-e077-4e69-a595-d3e7754fcda0" />

Each class can be identified through specific characteristics in the OCT images.

| Class  | Full Name                        | OCT Finding (Main Characteristics)                                                               |
| ------ | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| AMD    | Age-related Macular Degeneration | Damage in the macular region, retinal changes, fluid accumulation, and age-related abnormalities |
| CNV    | Choroidal Neovascularization     | Abnormal blood vessel growth under the retina, fluid leakage, and retinal layer thickening       |
| CSR    | Central Serous Retinopathy       | Accumulation of subretinal fluid and retinal layer separation                                    |
| DME    | Diabetic Macular Edema           | Macular thickening caused by fluid accumulation due to diabetes                                  |
| DR     | Diabetic Retinopathy             | Retinal vascular damage caused by diabetes, hemorrhages, and structural changes                  |
| Drusen | Drusen deposits                  | Yellow deposits beneath the retina (between RPE and Bruch's membrane)                            |
| MH     | Macular Hole                     | Formation of a gap or hole in the central macular region                                         |
| Normal | Healthy Retina                   | Normal retinal layer structure without pathological abnormalities                                |



## Data Preprocessing & Augmentation 

The complete data augmentation process has already been performed on the dataset.
Techniques such as cropping, padding, and horizontal flipping were used to increase the size of the training set and reduce overfitting.
Therefore, no additional augmentation is applied during training.
The only preprocessing steps applied are:
Resizing all images to 224 × 512.
Normalizing the images before feeding them into the model.
Normalization helps improve training stability, accelerates convergence, and reduces large fluctuations in gradient updates during optimization.

# Vision Transformer Architecture

One of the main limitations of Convolutional Neural Networks (CNNs) is that each convolutional layer only processes a local region of the image. Since convolutional filters have limited receptive fields, they can only observe a small part of the image at each step.
Therefore, if the model needs to understand the relationship between two distant regions of an image, it has to pass through many convolutional layers before those regions can influence each other.
To overcome this limitation, Vision Transformers (ViT) are used in this project.

Vision Transformer was first introduced in the paper: **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)**

and created a major shift in Computer Vision by applying the Transformer architecture, originally developed for Natural Language Processing, directly to image understanding tasks.
Instead of processing an image using convolution operations, Vision Transformers divide the image into smaller image patches. Each patch is treated similarly to a word token in NLP.
In this approach, every image patch can directly establish relationships with all other patches from the beginning of the network through the self-attention mechanism. Therefore, the model does not need many convolutional layers to gradually learn relationships between distant image regions.

<img width="2720" height="3720" alt="vit_full_architecture_with_encoder_zoom" src="https://github.com/user-attachments/assets/bd1dc0ff-a465-4e43-ac1b-db3e4fc40679" />


## Model Pipeline

The complete Vision Transformer pipeline consists of the following stages:

Patch Embedding
Adding CLS Token
Position Embedding
Multi-Head Self-Attention
MLP Block
Transformer Encoder Blocks:
Classification Head

## Patch Embedding

Unlike CNNs, images are not directly processed by Vision Transformers.
Before entering the Transformer Encoder, the image must be converted into a sequence of tokens.

First, the image is divided into small patches of size 16 × 16.
Each patch contains: 16 x 16 = 256 pixel values.

Since all images are resized during preprocessing to:
the total number of patches is:

$$ \frac{224}{16} \times \frac{512}{16}=14 \times 32=448 $$

Therefore, each image is converted into 448 patch tokens.

Instead of manually extracting these patches, a 2D convolution layer is used.
By setting:  Kernel size = 16   and   Stride = 16
the convolution layer automatically divides the image into non-overlapping patches.
The output contains 448 patch embeddings, where each patch is represented by a feature vector with 256 dimensions.


## CLS Token 

The CLS Token is an additional learnable token that is added at the beginning of the patch sequence.
Its purpose is to represent the overall information of the entire image.
During training, the CLS token learns to aggregate useful information from all image patches through the attention mechanism.
At the final stage of the network, instead of using all patch representations, only the CLS token representation is used for classification.

## Position Embedding

The Transformer architecture itself has no information about the spatial relationship between patches.
For example, the model does not know whether two patches are neighbors or located in completely different areas of the image.

To provide spatial information, a learnable Position Embedding vector is added to each patch token.
The input sequence becomes:
$$ Patch\ Embedding + Position\ Embedding $$

This allows the model to understand the position of each patch inside the original image.

## Multi-Head Self-Attention

Multi-Head Self-Attention is the core component of Vision Transformers.
In this mechanism, the model learns the relationship between every image patch and all other patches.
Unlike CNNs, where each region only interacts with its local neighborhood, the attention mechanism allows any patch to directly communicate with any other patch, even if they are far apart in the image.

For every token, three vectors are generated using learnable neural networks:

Query (Q)
Key (K)
Value (V)

Query (Q)

The Query represents what information this token is looking for from other tokens.

In other words:

What type of information does this patch need from other patches?

Key (K)

The Key represents the information that a token provides and determines whether this information can be useful for other tokens based on their Queries.

In other words:

What information does this patch contain that may be relevant to other patches?

Value (V)

The Value contains the actual information that will be transferred when the Key of a token matches the Query of another token.

In other words:

If another token pays attention to me, what information should I transfer?

The attention score between each token and other tokens is calculated using the following formula:

$$ Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d_k}})V $$

First, by multiplying:

$$ Q \times K^T $$

the similarity between the Query of each token and the Keys of all other tokens is calculated.
For example:
Query of Patch 1:
$$ Q=[1,0] $$

Patch 1:
$$ K_1=[1,0] $$

Patch 2:
$$ K_2=[0.8,0.2] $$

Patch 3:
$$ K_3=[0,1] $$

The similarity values are:

Similarity(Patch 1, K1):
$$ QK_1^T = [1,0] \times [1,0]^T = 1 $$


Similarity(Patch 1, K2):
$$
QK_2^T = [1,0] \times [0.8,0.2]^T = 0.8
$$


Similarity(Patch 1, K3):
$$
QK_3^T = [1,0] \times [0,1]^T = 0
$$

Therefore, Patch 1 pays the most attention to Patch 1 itself, then to Patch 2, and has no attention to Patch 3.

Then, the obtained similarity scores are divided by the square root of the Key dimension:
$$
\sqrt{d_k}
$$
and passed through the Softmax function.
This step normalizes the values.

In the final step, the calculated attention weights are applied to the Value vectors.

The updated representation of Patch 1 is calculated as:

$$
V(Patch1) + 
W_1 \times V_1 +
W_2 \times V_2 +
W_3 \times V_3
$$

Therefore, the representation of Patch 1 is updated, and it now contains information from other patches according to the amount of attention it has assigned to them.
The information from other patches influences the representation of Patch 1 based on their importance.

The Attention mechanism is implemented using **Multi-Head** Attention.
This means that instead of performing attention on the entire embedding vector at once, the embedding dimension is divided into multiple smaller vectors called attention heads.
In this project, the patch embedding dimension is 256 and it is divided into 8 different attention heads.
Therefore, each attention head operates on:
$$
\frac{256}{8}=32
$$
features.

Each attention head can learn different types of relationships between image patches.
For example, one head may focus on structural changes in retinal layers, while another head may learn relationships related to fluid accumulation or abnormal regions.

All Query, Key, and Value vectors, as well as the attention patterns learned by different heads, are automatically learned by neural networks during training.
We do not manually define which features the model should focus on or how much attention each patch should give to other patches.
Instead, the Vision Transformer learns these relationships automatically through the training process.

## MLP Block

After the Attention mechanism has learned the relationships between image patches, the model needs to understand what these relationships mean when combined together.
For example, after the Multi-Head Self-Attention stage, the information collected by a patch may contain different features from other patches:

- A patch may focus on fluid accumulation in the lower retinal layers.
- Another patch may provide information about retinal layer thickening.
- Another patch may capture overall structural changes in the retina.

The MLP Block receives these updated patch representations and learns how these extracted features can be combined to recognize higher-level patterns related to different diseases.
Therefore, each token representation is processed independently through a feed-forward neural network to learn more complex feature representations.

In this project, the MLP Block first expands the embedding dimension:
$$
256 \rightarrow 1024
$$
and after applying the activation function and processing the features, it projects them back to the original dimension:
$$
1024 \rightarrow 256
$$
This allows the model to learn richer and more complex representations while keeping the original token dimension for the next stages.

## Transformer Encoder Block

Each Transformer Encoder Block is the main building unit of the Vision Transformer architecture.
Its purpose is to enrich the representation of each image patch by repeatedly applying Attention and Feature Transformation operations.

The two main components: **Multi-Head Self-Attention** and **MLP Block** are placed sequentially inside each Transformer Encoder Block.
Multiple Transformer Encoder Blocks can be stacked together to increase the model capacity.

In this project, the number of Transformer Encoder Blocks is set to 6.

The input tokens generated from the previous stages are passed through these encoder blocks, where they are progressively transformed into more meaningful feature representations.

So In each Transformer Encoder Block we have : 

Input Tokens
Layer Normalization
Multi-Head Self-Attention
Residual Addition
Layer Normalization
MLP Block
Residual Addition
Output Tokens

Expalining Unknown Parts:

**Layer Normalization**

Layer Normalization normalizes the values inside the network and helps stabilize the training process.

It improves gradient flow and makes optimization easier, especially in deep Transformer architectures.

**Residual Addition**

Instead of replacing the input with the output of each sub-layer, the original input is added back to the result.

This technique helps preserve information and improves gradient propagation during training.

The equations are:

$$
x = x + Attention(Norm_1(x))
$$

and

$$
x = x + MLP(Norm_2(x))
$$

Residual connections help prevent the vanishing gradient problem and allow deeper Transformer networks to be trained effectively.

## So We Have the Full Vision Transformer Model

After implementing all the individual components, the complete Vision Transformer architecture is constructed by connecting all modules together in a single model.

The architecture consists of:

- Patch Embedding
- CLS Token
- Position Embedding
- Multiple Transformer Encoder Blocks
- Layer Normalization
- Classification Head

The number of Transformer Encoder Blocks can be adjusted depending on the required model capacity.
In this project, the number of Transformer Encoder Blocks is set to 6

**To sum up, this is the complete workflow performed by our Vision Transformer model, from receiving the input retinal OCT image to generating the final disease classification result :**

1. The input image is first converted into patch tokens through the Patch Embedding layer.

2. Then, the CLS Token is added to the beginning of the patch sequence, and Position Embeddings are added to provide spatial information.

3. The resulting sequence is passed through multiple Transformer Encoder Blocks, where the model gradually learns complex relationships between different regions of the retinal OCT images.

4. After passing through all Transformer Encoder Blocks, a final Layer Normalization operation is applied to stabilize the output representations.

5. At the classification stage, only the representation of the CLS Token is extracted. The CLS Token contains the aggregated information from all image patches and represents the overall image features.
   
$$
CLS_{output}=x[:,0]
$$

6. This representation is then passed to a simple fully connected neural network with 8 outputs, corresponding to the eight retinal OCT classes in the dataset.
The final output represents the predicted probability distribution over all classes.
During training, the model uses **CrossEntropyLoss** to compare the predicted probabilities with the true labels
The class with the highest predicted probability is selected as the final classification result.

