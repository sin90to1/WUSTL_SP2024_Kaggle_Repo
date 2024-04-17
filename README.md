# README

Competition URL: https://www.kaggle.com/competitions/applications-of-deep-learning-wustl-spring-2024/overview

Team members: Bojia Shi; Alyssa Ho; Mengxiao Li

Team name: MBTMan

In this competition, we have tested on several models:

* Visual Transformer
* ResNet-34
* ResNet-18 

ResNet-34 has the best result, with a final score of around 5.7 after 3 submissions.

## Project Structure

├── best_model.pth: The trained best model. Here it is ResNet-34

├── resnet_18.py: ResNet-18 model

├── resnet_34.py: ResNet-34 model (Our final solution)

├── data_augmentation.py: Data augmentation based on age distribution in the training set

├── solution.ipynb: ViT (Test version)

├── vit.ipynb & vit.py: Visual Transformer (Pretrained version)

├── test.py: Test the best model and output the result to the submission.csv file

├── submission.csv: The CSV file for submission

└── README.md

