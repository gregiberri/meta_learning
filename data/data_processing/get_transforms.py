
from torchvision import transforms

from data.data_processing.rotate_image import rotate_image


def get_transforms_for_dataset(dataset_name, args, k):
    if "cifar10" in dataset_name or "cifar100" in dataset_name:
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(args.classification_mean, args.classification_std)]

        transform_evaluate = [
            transforms.ToTensor(),
            transforms.Normalize(args.classification_mean, args.classification_std)]

    elif 'omniglot' in dataset_name:

        transform_train = [rotate_image(k=k, channels=args.image_channels), transforms.ToTensor()]
        transform_evaluate = [transforms.ToTensor()]


    elif 'imagenet' in dataset_name:

        transform_train = [transforms.Compose([

            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])]

        transform_evaluate = [transforms.Compose([

            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])]

    return transform_train, transform_evaluate
