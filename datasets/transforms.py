from torchvision import transforms

def get_transform(train):
    transforms_list =[]
    transforms_list.append(transforms.Resize((512,512)))
    transforms_list.append(transforms.ToTensor())
    
    if train:
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        #transforms_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3))
        transforms_list.append(transforms.RandomRotation(degrees=10))
    return transforms.Compose(transforms_list)