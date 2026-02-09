_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/ext_cls/cls_city_scapes.txt',
    concept_path='./configs/concept_bank/cb_sam3_ns.pt',
    concept_user='city_scapes',
    confidence_threshold=0.3,
)

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = './data/cityscapes'

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
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
