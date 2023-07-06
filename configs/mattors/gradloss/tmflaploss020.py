# model settings
model = dict(
    type='IndexNet',
    backbone=dict(
        type='DeeplabEncoderDecoder',
        encoder=dict(type='ResEncoder', backbone='resnet50s',
            outstride=16,in_channels=6, freeze_bn=False),
        decoder=dict(type='TMFdecoder')),
    loss_alpha=dict(type='CharbonnierLoss', loss_weight=0.5, sample_wise=True),
    loss_comp=dict(
        type='CharbonnierCompLoss', loss_weight=1.5, sample_wise=True),
    laploss_alpha=dict(type='LapLoss',loss_weight=0.2),
    pretrained=None)
# model training and testing settings
# model training and testing settings
train_cfg = dict(train_backbone=True)
test_cfg = dict(metrics=['SAD', 'MSE', 'GRAD', 'CONN'])

# dataset settings
dataset_type = 'AdobeComp1kDataset'
data_root = './data/adobe_composition-1k/'
bg_dir = './data/coco/train2017'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', key='alpha', flag='grayscale'),
    dict(type='LoadImageFromFile', key='fg'),
    dict(type='RIRandomLoadResizeBg', bg_dir=bg_dir),
    dict(
        type='CompositeFg',
        fg_dirs=[
            data_root + 'Training_set/Adobe-licensed images/fg',
            data_root + 'Training_set/Other/fg'
        ],
        alpha_dirs=[
            data_root + 'Training_set/Adobe-licensed images/alpha',
            data_root + 'Training_set/Other/alpha'
        ]),
    dict(type='RandomJitter'),
    dict(
        type='RIRandomAffine',
        keys=['alpha', 'fg'],
        degrees=30,
        scale=(0.8, 1.25),
        shear=10,
        flip_ratio=0.5),
    dict(type='GenerateTrimap', kernel_size=(1, 30)),
    dict(type='CropAroundCenter', crop_size=512),
    dict(type='MergeFgAndBg', save_original_img=True),
    dict(type='RescaleToZeroOne',  keys=['merged', 'alpha', 'ori_merged', 'fg', 'bg']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(type='Collect', keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg'], meta_keys=[]),
    dict(type='ImageToTensor', keys=['merged', 'alpha', 'trimap', 'ori_merged', 'fg', 'bg']),
    dict(type='FormatTrimap', to_onehot=True),
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='alpha',
        flag='grayscale',
        save_original_img=True),
    dict(
        type='LoadImageFromFile',
        key='trimap',
        flag='grayscale',
        save_original_img=True),
    dict(type='LoadImageFromFile', key='merged'),
    dict(type='Pad', keys=['trimap', 'merged'], mode='reflect'),
    dict(type='RescaleToZeroOne', keys=['merged']),
    dict(type='Normalize', keys=['merged'], **img_norm_cfg),
    dict(
        type='Collect',
        keys=['merged', 'trimap'],
        meta_keys=[
            'merged_path', 'pad', 'merged_ori_shape', 'ori_alpha', 'ori_trimap'
        ]),
    dict(type='ImageToTensor', keys=['merged', 'trimap']),
    dict(type='FormatTrimap', to_onehot=True),

]
data = dict(
    # train
    samples_per_gpu=16,
    workers_per_gpu=16,
    drop_last=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'training_list.json',
        data_prefix=data_root,
        pipeline=train_pipeline),
    # validation
    val_samples_per_gpu=1,
    val_workers_per_gpu=4,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline),
    # test
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_list.json',
        data_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizers = dict(
    constructor='DefaultOptimizerConstructor',
    type='Adam',
    lr=1e-2,
    paramwise_cfg=dict(custom_keys={'encoder.layers': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001)

# checkpoint saving
checkpoint_config = dict(interval=2000, by_epoch=False)
evaluation = dict(interval=2000, save_image=False, gpu_collect=False)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='gca'))
    ])
# yapf:enable

# runtime settings
total_iters = 200000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/gca'
load_from = None
resume_from = None
workflow = [('train', 1)]
