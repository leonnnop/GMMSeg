_base_ = [
    '../../models/segformer_mit-b0-gmm.py', '../../../_base_/datasets/cityscapes_768x768.py',
    '../../../_base_/default_runtime.py', '../../../_base_/schedules/schedule_80k.py'
]

model = dict(
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)),
    pretrained='pretrain/mit_b5_mmseg.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        channels=768,
        num_classes=19,
        in_channels=[64, 128, 320, 512],
        decoder_params=dict(
            # * basic prototype setup
            factor_n=1,
            num_prototype=3,
            embed_dim=64,),
        ),
)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
