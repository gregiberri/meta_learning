from data.data_processing.get_transforms import get_transforms_for_dataset


def augment_image(image, k, channels, augment_bool, config, dataset_name):
    transform_train, transform_evaluation = get_transforms_for_dataset(dataset_name=dataset_name, args=config, k=k)
    if len(image.shape) > 3:
        images = [item for item in image]
        output_images = []
        for image in images:
            if augment_bool is True:
                for transform_current in transform_train:
                    image = transform_current(image)
            else:
                for transform_current in transform_evaluation:
                    image = transform_current(image)
            output_images.append(image)
        image = torch.stack(output_images)
    else:
        if augment_bool is True:
            # meanstd transformation
            for transform_current in transform_train:
                image = transform_current(image)
        else:
            for transform_current in transform_evaluation:
                image = transform_current(image)
    return image
