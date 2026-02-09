_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/ext_cls/cls_ade20k.txt',
    concept_path='./configs/concept_bank/cb_sam3_ns.pt',
    concept_user='ade20k',
    confidence_threshold=0.3,
)

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = './data/ade/ADEChallengeData2016'

test_pipeline = [
    dict(type='LoadPILFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
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
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
