#!/bin/bash
source ./pipeline_utils.sh

# 1D Base
run_full_pipeline "$TOK_1D" "1D-Baseline" \
    ae_no_label=True \
    phases=1500000:DO_L1-DO_LPIPS-DO_GAN_G,DO_GAN_D:1,1

# 2D Base
run_full_pipeline "$TOK_2D" "2D-Baseline" \
    ae_no_label=True \
    phases=1500000:DO_L1-DO_LPIPS-DO_GAN_G,DO_GAN_D:1,1


# Prologue Base
run_full_pipeline "$TOK_Prologue" stages=1,2 "Prologue-B-Tokenizer" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    stage1.ARModel.tied_embedding=False \
    stage1.ARModel.layer_num=7 \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    stage1.label_drop_prob=1.0

# B-B
AR=$AR_B run_full_pipeline "$TOK_Prologue" stages=3 "Prologue-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=$(find_best_ckpt "Prologue-B-Tokenizer")

# B-L
AR=$AR_L run_full_pipeline "$TOK_Prologue" stages=3 "Prologue-B-L" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=$(find_best_ckpt "Prologue-B-Tokenizer")

# B-XL
AR=$AR_XL run_full_pipeline "$TOK_Prologue" stages=3 "Prologue-B-XL" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    tokenizer_ckpt_path=$(find_best_ckpt "Prologue-B-Tokenizer")

# Prologue Large
AR_JOINT=$AR_S run_full_pipeline "$TOK_Prologue" stages=1,2 "Prologue-L-Tokenizer" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    stage1.ARModel.tied_embedding=False \
    stage1.ARModel.layer_num=7 \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    stage1.label_drop_prob=1.0 \
    perceptual_network=convnext \
    codebook_size=4096 \
    Decoder.dim=1024 \
    Decoder.layer_num=24 \
    Decoder.heads=16


# L-B
AR=$AR_B run_full_pipeline "$TOK_Prologue" stages=3 "Prologue-L-B" \
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
    tokenizer_ckpt_path=$(find_best_ckpt "Prologue-L-Tokenizer")


# L-L
AR=$AR_L run_full_pipeline "$TOK_Prologue" stages=3 "Prologue-L-L" \
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
    tokenizer_ckpt_path=$(find_best_ckpt "Prologue-L-Tokenizer")

# L-XL
AR=$AR_XL run_full_pipeline "$TOK_Prologue" stages=3 "Prologue-L-XL" \
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
    tokenizer_ckpt_path=$(find_best_ckpt "Prologue-L-Tokenizer")

# Prologue-Post B-B 
run_full_pipeline "$TOK_Prologue_Post" "Prologue-Post-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    stage1.ARModel.tied_embedding=False \
    stage1.ARModel.layer_num=7 \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    stage1.label_drop_prob=1.0

# Prologue-OneStage B-B
AR_JOINT=$AR_B run_full_pipeline "$TOK_Prologue" stages=1 "Prologue-OneStage-B-B" \
    prior_enc_semantic_weight=3.0 \
    prior_enc_visual_weight=3.0 \
    ae_no_label=True \
    ARModel.ste_ar_embedding=True \
    SemanticQuantizer.temperature=0.1 \
    use_eos=True \
    prior_visual_dropout=0.5 \
    stage1.label_drop_prob=0.1