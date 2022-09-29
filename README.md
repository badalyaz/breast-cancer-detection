## Goal
Our goal is to create an object detection, classification and segmentation model that will be able to detect, classify and segment breast tumors from X-ray images. We believe that with our objective reached many lives will be saved. We use several Machine Learning techniques to approach our goal.

## Sample test
We put a small sample of images that one can experiment with.
- In the folder HeatmapGenerator/Sample_data/`Images`, there are 19 sample images for experiments.
- In the folder named HeatmapGenerator/Sample_data/`generated_heatmaps`, there are generated and saved heatmaps for 316 images.

In folder `Notebook` there are jupyter notebook files containing visualizations and evaluations.
- There is a file named [Check heatmaps.ipynb](https://gitlab.com/sven.badalyan/breast_cancer_gitlab/-/blob/workers_branch/HeatmapGenerator/Notebooks/Check%20heatmaps.ipynb) in the HeatmapGenerator folder, with which you can check how the Heatmap generator works.
- [Threshold results.ipynb](https://gitlab.com/sven.badalyan/breast_cancer_gitlab/-/blob/workers_branch/HeatmapGenerator/Notebooks/Threshold%20results.ipynb) is the file where we tune thresholds for density calculation
- [Evaluate threshold results.ipynb](https://gitlab.com/sven.badalyan/breast_cancer_gitlab/-/blob/workers_branch/HeatmapGenerator/Notebooks/Evaluate%20threshold%20results%20with%20bins.ipynb) is the file where we tune thresholds 
- [Evaluate threshold results with bins.ipynb](https://gitlab.com/sven.badalyan/breast_cancer_gitlab/-/blob/workers_branch/HeatmapGenerator/Notebooks/Evaluate%20threshold%20results%20with%20bins.ipynb) is the file where we tune thresholds using bins

There are also files in which one can perform Grid Search in different ways (without bins, with bins, etc.).

## Example
Here is an example of the heatmaps generated by heatmap generator
![Examples of heatmaps generated by Heatmap generator](https://github.com/badalyaz/cancer_detection/blob/interns_branch/HeatmapGenerator/heatmaps.png "heatmaps")

## Technical details
Our first approach was taking the MASK R-CNN Segmentation model and replacing its backbone with the ResNet-22 pre-trained on InBreast mammographic dataset. The backbone and the heatmap generator we modified and used can be found at [Breast Cancer Classifier](https://github.com/nyukat/breast_cancer_classifier) GitHub Repository.

Our next approach was using the [F-RCNN CAD](https://github.com/riblidezso/frcnn_cad) model to get a more robust model. The main hardship we faced was installing Caffe, which we eventually successfully did, in steps described in [this](https://github.com/badalyaz/cancer_detection/blob/interns_branch/Documents/Installing%20caffe.pdf) file.

Our final approach is the one described in this Repository.

## Datasets
- [InBreast](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset) ~400 .dcm images, 91 of which have extracted [bounding boxes](https://github.com/riblidezso/frcnn_cad/blob/master/data/inbreast_ground_truth_cancer_bbox_rois.tsv)
- [InBreast](https://www.dropbox.com/sh/eu7wc3hl30a6knt/AABhn6BmENJFo-5Ya0wEwvQCa?dl=0) ~90 .png images with bounding boxes and masks
- [DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) .tcia file containing ~3500 .dcm images with bounding boxes and masks for the majority of the images
- [SFUniversity](http://www.eng.usf.edu/cvprg/Mammography/Database.html) ~8400 .ljpeg images
- [DBT](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580) .dcm images, each consisting of ~60 frames

## Annotations
Processed annotations of the above datasets.
- [DDSM](https://github.com/badalyaz/cancer_detection/tree/interns_branch/DataProcessing/Annotations/DDSMAnnotations)
- [InBreast](https://github.com/badalyaz/cancer_detection/tree/interns_branch/DataProcessing/Annotations/INbreastAnnotations)
- [SFUniversity](https://github.com/badalyaz/cancer_detection/blob/interns_branch/DataProcessing/Annotations/SFUAnnotations.pickle)

## Reference
@article{wu2019breastcancer, 
    title = {Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening},
    author = {Nan Wu and Jason Phang and Jungkyu Park and Yiqiu Shen and Zhe Huang and Masha Zorin and Stanis\l{}aw Jastrz\k{e}bski and Thibault F\'{e}vry and Joe Katsnelson and Eric Kim and Stacey Wolfson and Ujas Parikh and Sushma Gaddam and Leng Leng Young Lin and Kara Ho and Joshua D. Weinstein and Beatriu Reig and Yiming Gao and Hildegard Toth and Kristine Pysarenko and Alana Lewin and Jiyon Lee and Krystal Airola and Eralda Mema and Stephanie Chung and Esther Hwang and Naziya Samreen and S. Gene Kim and Laura Heacock and Linda Moy and Kyunghyun Cho and Krzysztof J. Geras}, 
    journal = {IEEE Transactions on Medical Imaging},
    year = {2019}
}