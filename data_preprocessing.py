from utils.data_loader import create_datasets

# Пути к данным
train_lr = "data/train/low_resolution"
train_hr = "data/train/high_resolution"
valid_lr = "data/valid/low_resolution"
valid_hr = "data/valid/high_resolution"

# Создание датасетов
train_ds, valid_ds = create_datasets(
    train_lr_dir=train_lr,
    train_hr_dir=train_hr,
    valid_lr_dir=valid_lr,
    valid_hr_dir=valid_hr,
    patch_size=32,
    scale=2,
    normalize='0-1'
)