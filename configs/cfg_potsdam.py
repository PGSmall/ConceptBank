_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/ext_cls/cls_potsdam.txt',
    concept_path='./configs/concept_bank/cb_sam3_rs.pt',
    concept_user='potsdam',
    bg_idx=5,
    bg_thr=0.0,
    calib_per_class=50,
    calib_max_class=100,
    confidence_threshold=0.1,
)

# dataset settings
dataset_type = 'PotsdamDataset'
data_root = './data/potsdam'

test_pipeline = [
    dict(type='LoadPILFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackPILInputs', to_tensor=False)
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))