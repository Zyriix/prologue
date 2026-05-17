#!/bin/bash
source ./pipeline_utils.sh

# 1D Base

# Paper CFG : overall-cosine, scale=19, power=1.75 → gFID=6.10
sweep_cfg "$TOK_1D" "1D-Baseline" \
    ae_no_label=True \
    tokenizer_ckpt_path=ckpts/1d-tokenizer \
    ar_ckpt_path=ckpts/ar-1d-base \
    cfg_grid="oc:19:1.75"

# Paper noCFG : temperature=0.85 → gFID_noCFG=19.32
sweep_cfg "$TOK_1D" "1D-Baseline" \
    ae_no_label=True \
    tokenizer_ckpt_path=ckpts/1d-tokenizer \
    ar_ckpt_path=ckpts/ar-1d-base \
    cfg_grid="nt:0.85"


# 2D Base

# Paper CFG : overall-cosine, scale=10, power=1.5 → gFID=5.02
sweep_cfg "$TOK_2D" "2D-Baseline" \
    ae_no_label=True \
    tokenizer_ckpt_path=ckpts/2d-tokenizer \
    ar_ckpt_path=ckpts/ar-2d-base \
    cfg_grid="oc:10:1.5"

# Paper noCFG : temperature=0.85 → gFID_noCFG=21.01
sweep_cfg "$TOK_2D" "2D-Baseline" \
    ae_no_label=True \
    tokenizer_ckpt_path=ckpts/2d-tokenizer \
    ar_ckpt_path=ckpts/ar-2d-base \
    cfg_grid="nt:0.85"


# B-B
AR=$AR_B sweep_cfg "$TOK_Prologue" "Prologue-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=ckpts/prologue-b-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-b-b \
    pretoken_dir=ckpts/prologue-b-tokenizer/pretoken \
    cfg_grid="sc:0.7:3.75:0.2"

AR=$AR_B sweep_cfg "$TOK_Prologue" "Prologue-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=ckpts/prologue-b-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-b-b \
    pretoken_dir=ckpts/prologue-b-tokenizer/pretoken \
    cfg_grid="nt2:0.7:0.9"


# B-L
AR=$AR_L sweep_cfg "$TOK_Prologue" "Prologue-B-L" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=ckpts/prologue-b-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-b-l \
    pretoken_dir=ckpts/prologue-b-tokenizer/pretoken \
    cfg_grid="sc:0.8:3.0:0.225"

AR=$AR_L sweep_cfg "$TOK_Prologue" "Prologue-B-L" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=ckpts/prologue-b-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-b-l \
    pretoken_dir=ckpts/prologue-b-tokenizer/pretoken \
    cfg_grid="nt2:0.7:0.9"


# B-XL
AR=$AR_XL sweep_cfg "$TOK_Prologue" "Prologue-B-XL" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=ckpts/prologue-b-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-b-xl \
    pretoken_dir=ckpts/prologue-b-tokenizer/pretoken \
    cfg_grid="sc:0.8:2.75:0.25"

AR=$AR_XL sweep_cfg "$TOK_Prologue" "Prologue-B-XL" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=ckpts/prologue-b-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-b-xl \
    pretoken_dir=ckpts/prologue-b-tokenizer/pretoken \
    cfg_grid="nt2:0.8:0.9"


# L-B
AR=$AR_B sweep_cfg "$TOK_Prologue" "Prologue-L-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    perceptual_network=convnext \
    codebook_size=4096 \
    Decoder.dim=1024 \
    Decoder.layer_num=24 \
    Decoder.heads=16 \
    eval_batch_size=400 \
    tokenizer_ckpt_path=ckpts/prologue-l-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-l-b \
    pretoken_dir=ckpts/prologue-l-tokenizer/pretoken \
    cfg_grid="sc:0.65:2.75:0.25"

AR=$AR_B sweep_cfg "$TOK_Prologue" "Prologue-L-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    perceptual_network=convnext \
    codebook_size=4096 \
    Decoder.dim=1024 \
    Decoder.layer_num=24 \
    Decoder.heads=16 \
    eval_batch_size=400 \
    tokenizer_ckpt_path=ckpts/prologue-l-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-l-b \
    pretoken_dir=ckpts/prologue-l-tokenizer/pretoken \
    cfg_grid="nt2:0.8:0.95"


# L-L
AR=$AR_L sweep_cfg "$TOK_Prologue" "Prologue-L-L" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    perceptual_network=convnext \
    codebook_size=4096 \
    Decoder.dim=1024 \
    Decoder.layer_num=24 \
    Decoder.heads=16 \
    eval_batch_size=400 \
    tokenizer_ckpt_path=ckpts/prologue-l-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-l-l \
    pretoken_dir=ckpts/prologue-l-tokenizer/pretoken \
    cfg_grid="sc:0.7:2.5:0.25"

AR=$AR_L sweep_cfg "$TOK_Prologue" "Prologue-L-L" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    perceptual_network=convnext \
    codebook_size=4096 \
    Decoder.dim=1024 \
    Decoder.layer_num=24 \
    Decoder.heads=16 \
    eval_batch_size=400 \
    tokenizer_ckpt_path=ckpts/prologue-l-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-l-l \
    pretoken_dir=ckpts/prologue-l-tokenizer/pretoken \
    cfg_grid="nt2:0.95:0.9"


# L-XL
AR=$AR_XL sweep_cfg "$TOK_Prologue" "Prologue-L-XL" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    perceptual_network=convnext \
    codebook_size=4096 \
    Decoder.dim=1024 \
    Decoder.layer_num=24 \
    Decoder.heads=16 \
    eval_batch_size=800 \
    tokenizer_ckpt_path=ckpts/prologue-l-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-l-xl \
    pretoken_dir=ckpts/prologue-l-tokenizer/pretoken \
    cfg_grid="sc:0.7:2.25:0.225"

AR=$AR_XL sweep_cfg "$TOK_Prologue" "Prologue-L-XL" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    perceptual_network=convnext \
    codebook_size=4096 \
    Decoder.dim=1024 \
    Decoder.layer_num=24 \
    Decoder.heads=16 \
    eval_batch_size=800 \
    tokenizer_ckpt_path=ckpts/prologue-l-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-l-xl \
    pretoken_dir=ckpts/prologue-l-tokenizer/pretoken \
    cfg_grid="nt2:1.0:0.9"


# Prologue-Post B-B
sweep_cfg "$TOK_Prologue_Post" "Prologue-Post-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=ckpts/prologue-post-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-post-b \
    pretoken_dir=ckpts/prologue-post-tokenizer/pretoken \
    cfg_grid="sc:0.6:3.75:0.25"

sweep_cfg "$TOK_Prologue_Post" "Prologue-Post-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=ckpts/prologue-post-tokenizer \
    ar_ckpt_path=ckpts/ar-prologue-post-b \
    pretoken_dir=ckpts/prologue-post-tokenizer/pretoken \
    cfg_grid="nt2:0.8:0.8"


# Prologue-OneStage B-B
AR_JOINT=$AR_B sweep_cfg "$TOK_Prologue" "Prologue-OneStage-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    ema_reconstruction=False \
    ar_stage=1 \
    tokenizer_ckpt_path=ckpts/prologue-onestage-joint \
    cfg_grid="sc:0.8:3.75:0.25"

AR_JOINT=$AR_B sweep_cfg "$TOK_Prologue" "Prologue-OneStage-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    ema_reconstruction=False \
    ar_stage=1 \
    tokenizer_ckpt_path=ckpts/prologue-onestage-joint \
    cfg_grid="nt2:0.8:0.8"
