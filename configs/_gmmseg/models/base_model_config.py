find_unused_parameters = True
model = dict(
    decode_head=dict(
        decoder_params=dict(
            # * basic setup
            embed_dim=64,
            num_components=5,
            gamma=[0.999,0],
            # * sinkhorn
            factor_n=1,
            factor_c=1,
            factor_p=1,
            # *
            mem_size=32000,
            max_sample_size=20,
            # *
            update_GMM_interval=5,
        )
    ),
    train_cfg=dict(
        contrast_loss=True,
        contrast_loss_weight=0.01,
        sampler_mode='gmmseg',
    )
)
