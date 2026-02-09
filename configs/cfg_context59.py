_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/ext_cls/cls_context59.txt',
    concept_path='./configs/concept_bank/cb_sam3_ns.pt',
    concept_user='context59',
    confidence_threshold=0.3,
)

# dataset settings
dataset_type = 'PascalContext59Dataset'
data_root = './data/VOCdevkit/VOC2010'

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
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))
