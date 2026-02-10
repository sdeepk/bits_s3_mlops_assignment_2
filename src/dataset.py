from torchvision import datasets, transforms


def get_datasets(data_dir):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val", transform=val_tf)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test", transform=val_tf)

    return train_ds, val_ds, test_ds
