from models import NetHSP_GIN, Net_DRew_GIN, NetGIN, NetGCN, NetGAT


def get_model(args, device="cpu", num_features=None, num_classes=None):
    if args.model == "GIN":
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        if args.dataset.endswith("Prox"):
            emb_input = 10
        else:
            emb_input = -1
        model = NetGIN(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            eps=args.eps,
            train_eps=False,
            emb_input=emb_input,
        )
        return model
    elif args.model == "GIN-EPS":
        if args.dataset.endswith("Prox"):
            emb_input = 10  # The number of colors in the proximity dataset
        else:
            emb_input = -1
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        model = NetGIN(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            eps=args.eps,
            train_eps=True,
            emb_input=emb_input,
        )
        return model
    elif args.model == "GCN":
        if args.dataset.endswith("Prox"):
            emb_input = 10
        else:
            emb_input = -1
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        model = NetGCN(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            emb_input=emb_input,
        )
        return model

    elif args.model == "GAT":
        if args.dataset.endswith("Prox"):
            emb_input = 10
        else:
            emb_input = -1
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        model = NetGAT(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            emb_input=emb_input,
        )
        return model

    model_type, inside, outside = args.model.lower().split("_")

    if not inside in ["attn_nh", "global_attn_nh", "sum", "rsum", "edgesum"]:
        raise ValueError("Invalid inside model aggregation.")

    if not outside in ["sum", "weight", "eps_weight"]:
        raise ValueError("Invalid outside model aggregation.")

    emb_sizes = [args.emb_dim] * (args.num_layers + 1)
    if args.dataset.startswith("ogbg"):
        ogb_gc = args.dataset
    else:
        ogb_gc = None

    # DRew-GIN
    if model_type == 'drew' or model_type=='share_drew':
        if model_type.startswith('share'):
            print('Using DRew-GIN model with weight sharing')
            use_weight_share = True
        else:
            use_weight_share = False
            print('Using DRew-GIN model')

        model = Net_DRew_GIN(
            args.nu,
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            max_distance=args.max_distance,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            inside_aggr=inside,
            outside_aggr=outside,
            mode=args.mode,
            eps=args.eps,
            ogb_gc=ogb_gc,
            batch_norm=args.batch_norm,
            layer_norm=args.layer_norm,
            pool_gc=args.pool_gc,
            residual_frequency=args.res_freq,
            dataset=args.dataset,
            learnable_emb=args.learnable_emb,
            use_feat=args.use_feat,
            use_weight_share=use_weight_share,
        ).to(device)

    else:
        print("Model type is %s, defaulting to NetHSP_GIN" % args.model)
        model = NetHSP_GIN(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            max_distance=args.max_distance,
            scatter=args.scatter,
            drpt_prob=args.dropout,
            inside_aggr=inside,
            outside_aggr=outside,
            mode=args.mode,
            eps=args.eps,
            ogb_gc=ogb_gc,
            batch_norm=args.batch_norm,
            layer_norm=args.layer_norm,
            pool_gc=args.pool_gc,
            residual_frequency=args.res_freq,
            dataset=args.dataset,
            learnable_emb=args.learnable_emb,
            use_feat=args.use_feat,
        ).to(device)
    return model
