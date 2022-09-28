import random
from . import run_producer


#this function is used for producting heatmaps (original function from code)
def produce_heatmaps(image_array, model, device):


    parameters = dict(
        device_type="gpu",
        gpu_number=0,

        patch_size=256,

        stride_fixed=70,
        more_patches=5,
        minibatch_size=100,
        seed=0,

        initial_parameters="/home/server/other_projects/breast_cancer/HeatmapGenerator/models/sample_patch_model.p",
        input_channels=3,
        number_of_classes=4,

        heatmap_type=[0, 1],
        )

    random.seed(parameters['seed'])


    patches, case = run_producer.sample_patches_single(
        image_array[0],
        parameters=parameters,
    )

    all_prob = run_producer.get_all_prob(
        all_patches=patches,
        minibatch_size=parameters["minibatch_size"],
        model=model,
        device=device,
        parameters=parameters
    )

    heatmap_malignant, _ = run_producer.probabilities_to_heatmap(
        patch_counter=0,
        all_prob=all_prob,
        image_shape=case[0],
        length_stride_list=case[2],
        width_stride_list=case[1],
        patch_size=parameters['patch_size'],
        heatmap_type=parameters['heatmap_type'][0],
    )
    heatmap_benign, patch_counter = run_producer.probabilities_to_heatmap(
        patch_counter=0,
        all_prob=all_prob,
        image_shape=case[0],
        length_stride_list=case[2],
        width_stride_list=case[1],
        patch_size=parameters['patch_size'],
        heatmap_type=parameters['heatmap_type'][1],
    )


    return heatmap_benign, heatmap_malignant