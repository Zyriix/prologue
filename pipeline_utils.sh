#!/bin/bash
# Shared pipeline helpers; source this file from experiment scripts:
#   source ./pipeline_utils.sh
# Then call run_full_pipeline / sweep_cfg / find_best_ckpt / find_best_ar_ckpt.

# ============================================================================
# Multi-node / multi-GPU detection
# ============================================================================
if [[ -z "${GPUS_PER_NODE:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _CVD_ARR <<< "${CUDA_VISIBLE_DEVICES}"
    GPUS_PER_NODE="${#_CVD_ARR[@]}"
fi
: "${GPUS_PER_NODE:=8}"
: "${NUM_MACHINES:=${WORLD_SIZE:-1}}"
: "${MACHINE_RANK:=${RANK:-0}}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"

if (( NUM_MACHINES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" ]]; then
    echo "[error] multi-node requires MASTER_ADDR to be set" >&2
    exit 2
fi

ACCEL_LAUNCH=(accelerate launch
    --num_processes=$((GPUS_PER_NODE * NUM_MACHINES))
    --num_machines=${NUM_MACHINES}
    --machine_rank=${MACHINE_RANK}
    --main_process_ip=${MASTER_ADDR}
    --main_process_port=${MASTER_PORT}
)

echo "=== [pipeline] GPUs/node=${GPUS_PER_NODE} machines=${NUM_MACHINES} rank=${MACHINE_RANK} master=${MASTER_ADDR}:${MASTER_PORT} ==="

# ============================================================================
# Config layer paths (relative to the project root)
# ============================================================================
CFGD=${CFGD:-configs}

BASE=$CFGD/default.yaml
_TOK_DEFAULT=$CFGD/tokenizer/default.yaml
AR_DEFAULTS=$CFGD/ar/_defaults.yaml

TRAIN_AE=$CFGD/train/ae.yaml
TRAIN_AR=$CFGD/train/ar.yaml
TRAIN_EVAL_AE=$CFGD/train/eval_ae.yaml
TRAIN_EVAL_AR=$CFGD/train/eval_ar.yaml

# ============================================================================
# AR size variants (always paired with AR_DEFAULTS)
# ============================================================================
AR_S="$AR_DEFAULTS,$CFGD/ar/small.yaml"
AR_B="$AR_DEFAULTS,$CFGD/ar/base.yaml"
AR_L="$AR_DEFAULTS,$CFGD/ar/large.yaml"
AR_XL="$AR_DEFAULTS,$CFGD/ar/xlarge.yaml"

# AR size for standalone training (Phase 2+3: pretokenize + AR)
AR=${AR:-$AR_B}
# AR size for joint AE+AR training (Phase 1), typically smaller
AR_JOINT=${AR_JOINT:-$AR_S}

# Tokenizer presets: "model_configs[|phase1_overlay]"
TOK_2D="$_TOK_DEFAULT,$CFGD/tokenizer/2d.yaml"
TOK_1D="$_TOK_DEFAULT,$CFGD/tokenizer/1d.yaml"
TOK_Prologue="$_TOK_DEFAULT,$CFGD/tokenizer/prologue.yaml"
TOK_Prologue_Post="$_TOK_DEFAULT,$CFGD/tokenizer/prologue_post.yaml|$CFGD/train/post_overlay.yaml"

# ============================================================================
# Save / log directories (relative to the project root)
# ============================================================================
AE_SAVE_DIR=${AE_SAVE_DIR:-experiments}
AR_SAVE_DIR=${AR_SAVE_DIR:-experiments_ar}
AE_LOGDIR=${AE_LOGDIR:-logs/AE}
PRETOK_LOGDIR=${PRETOK_LOGDIR:-logs/Pretoken}
AR_LOGDIR=${AR_LOGDIR:-logs/AR}
EVAL_LOGDIR=${EVAL_LOGDIR:-logs/EVAL}
PRETOKEN_DIR=${PRETOKEN_DIR:-pretoken}

mkdir -p "$AE_LOGDIR" "$PRETOK_LOGDIR" "$AR_LOGDIR" "$EVAL_LOGDIR"

# ============================================================================
# Helper functions
# ============================================================================

# Newest run with the best rFID ckpt (fallback gFID).
find_best_ckpt() {
    local name=$1
    local exp_dirs
    exp_dirs=$(ls -d ${AE_SAVE_DIR}/${name}/*/ 2>/dev/null | sort -r)
    [ -z "$exp_dirs" ] && return 1
    local exp ckpt
    for exp in $exp_dirs; do
        ckpt=$(ls -d ${exp}ckpts/best-*-rFID=* 2>/dev/null \
            | sort -t'=' -k3 -g | head -n 1)
        if [ -z "$ckpt" ]; then
            ckpt=$(ls -d ${exp}ckpts/best-*-gFID=* 2>/dev/null \
                | sort -t'=' -k3 -g | head -n 1)
        fi
        [ -n "$ckpt" ] && echo "$ckpt" && return 0
    done
    return 1
}

# Newest run with the best no-CFG gFID ckpt (fallback last-*).
find_best_ar_ckpt() {
    local name=$1
    local exp_dirs
    exp_dirs=$(ls -d ${AR_SAVE_DIR}/${name}/*/ 2>/dev/null | sort -r)
    [ -z "$exp_dirs" ] && return 1
    local exp ckpt
    for exp in $exp_dirs; do
        ckpt=$(ls -d ${exp}ckpts/best-*-gFIDwoCFG=* 2>/dev/null \
            | sort -t'=' -k3 -g | head -n 1)
        if [ -z "$ckpt" ]; then
            ckpt=$(ls -d ${exp}ckpts/last-* 2>/dev/null \
                | sort -t'=' -k2 -g | tail -n 1)
        fi
        [ -n "$ckpt" ] && echo "$ckpt" && return 0
    done
    return 1
}

# resolve_pretoken_and_tok <tok_ckpt> -> "<pretoken_dir>|<resolved_tok_ckpt>"
# Looks up exact ckpt subdir -> last-* subdir -> legacy run-level npz layout.
resolve_pretoken_and_tok() {
    local tok_ckpt=$1
    local pretoken_base="${PRETOKEN_DIR}"

    local wandb_name run_id ckpt_name ae_ckpts_dir
    wandb_name=$(echo "$tok_ckpt" | sed 's|.*/experiments/||' | cut -d'/' -f1)
    run_id=$(echo   "$tok_ckpt" | sed 's|.*/experiments/[^/]*/||' | cut -d'/' -f1)
    ckpt_name=$(basename "$tok_ckpt")
    ae_ckpts_dir=$(dirname "$tok_ckpt")

    [ -z "$wandb_name" ] && return 1

    local pretoken_run_dir="${pretoken_base}/${wandb_name}/${run_id}"

    if [ -d "${pretoken_run_dir}/${ckpt_name}" ]; then
        echo "${pretoken_run_dir}/${ckpt_name}|${tok_ckpt}"
        return 0
    fi

    local last_pretoken
    last_pretoken=$(ls -d "${pretoken_run_dir}/last-"* 2>/dev/null \
        | sort -t'=' -k2 -g | tail -n1)
    if [ -n "$last_pretoken" ]; then
        local step last_tok=""
        step=$(basename "$last_pretoken" | grep -oP 'Step=\K[0-9]+' || true)
        if [ -n "$step" ] && [ -d "$ae_ckpts_dir" ]; then
            last_tok=$(ls -d "${ae_ckpts_dir}/last-"* 2>/dev/null \
                | grep "Step=${step}" | head -n1)
        fi
        [ -z "$last_tok" ] && last_tok="$tok_ckpt"
        echo "${last_pretoken}|${last_tok}"
        return 0
    fi

    if ls "${pretoken_run_dir}/"*.npz 2>/dev/null | head -n1 | grep -q .; then
        local last_tok=""
        if [ -d "$ae_ckpts_dir" ]; then
            last_tok=$(ls -d "${ae_ckpts_dir}/last-"* 2>/dev/null \
                | sort -t'=' -k2 -g | tail -n1)
        fi
        [ -z "$last_tok" ] && last_tok="$tok_ckpt"
        echo "${pretoken_run_dir}|${last_tok}"
        return 0
    fi

    return 1
}

_has_stage() { [[ ",${1}," == *",$2,"* ]]; }

# run_full_pipeline <tok_preset> [stages=1,2,3] <wandb_name> [extra_args...]
run_full_pipeline() {
    local tok=$1; shift

    # Split tok: "model_configs|phase1_overlay" or just "model_configs"
    local tok_configs="${tok%%|*}"
    local ae_overlay=""
    if [[ "$tok" == *"|"* ]]; then
        ae_overlay="${tok#*|}"
    fi

    local stages="1,2,3"
    if [[ "$1" == stages=* ]]; then
        stages="${1#stages=}"; shift
    fi
    local name=$1; shift

    local global_args=()
    local stage1_only=()
    local stage2_only=()
    local stage3_only=()
    for arg in "$@"; do
        case "$arg" in
            stage1.*) stage1_only+=("${arg#stage1.}") ;;
            stage2.*) stage2_only+=("${arg#stage2.}") ;;
            stage3.*) stage3_only+=("${arg#stage3.}") ;;
            *)        global_args+=("$arg") ;;
        esac
    done

    # Drop Phase-1-only keys for Phases 2/3.
    local ar_global_args=()
    local _user_ckpt=""
    for arg in "${global_args[@]}"; do
        case "$arg" in
            resume_enc=*|resume_dec=*|resume_ckpt_path=*|resume_gan=*|phases=*) ;;
            tokenizer_ckpt_path=*) _user_ckpt="${arg#tokenizer_ckpt_path=}" ;;
            *) ar_global_args+=("$arg") ;;
        esac
    done

    local common_sv="$BASE,$AR_JOINT,$tok_configs"
    local common_ar="$BASE,$AR,$tok_configs"
    local ae_configs="${common_sv},${TRAIN_AE}"
    [ -n "$ae_overlay" ] && ae_configs="${ae_configs},${ae_overlay}"
    echo "===== [${name}] stages=${stages} AR_JOINT=${AR_JOINT##*,} AR=${AR##*,} ====="

    if _has_stage "$stages" 1; then
        echo "===== [${name}] Phase 1: AE Training ====="
        "${ACCEL_LAUNCH[@]}" --mixed_precision=bf16 \
            train_tokenizer.py --configs=${ae_configs} \
            wandb_name=${name} \
            save_dir=${AE_SAVE_DIR}/${name} \
            "${global_args[@]}" "${stage1_only[@]}" \
            > ${AE_LOGDIR}/${name}.log 2>&1
        if [ $? -ne 0 ]; then
            echo "ERROR: [${name}] Phase 1 failed, skipping remaining phases."
            return 1
        fi
    else
        echo "===== [${name}] Phase 1: SKIPPED ====="
    fi

    if _has_stage "$stages" 2 || _has_stage "$stages" 3; then
        local ckpt="${_user_ckpt}"
        if [ -z "$ckpt" ]; then
            ckpt=$(find_best_ckpt ${name})
        fi
        if [ -z "$ckpt" ]; then
            echo "ERROR: No best checkpoint found for ${name}, skipping remaining phases."
            return 1
        fi
        echo "Best checkpoint: ${ckpt}"
    fi

    if _has_stage "$stages" 2; then
        echo "===== [${name}] Phase 2: Pretokenize ====="
        "${ACCEL_LAUNCH[@]}" \
            train_pretoken.py --configs=${common_ar},${TRAIN_AR} \
            wandb_name=${name}-Pretoken \
            tokenizer_ckpt_path=${ckpt} \
            "${ar_global_args[@]}" "${stage2_only[@]}" \
            > ${PRETOK_LOGDIR}/${name}.log 2>&1
        if [ $? -ne 0 ]; then
            echo "ERROR: [${name}] Phase 2 failed, skipping remaining phases."
            return 1
        fi
    else
        echo "===== [${name}] Phase 2: SKIPPED ====="
    fi

    if _has_stage "$stages" 3; then
        echo "===== [${name}] Phase 3: AR Training ====="
        "${ACCEL_LAUNCH[@]}" --mixed_precision=bf16 \
            train_ar.py --configs=${common_ar},${TRAIN_AR} \
            wandb_name=${name}-AR \
            tokenizer_ckpt_path=${ckpt} \
            save_dir=${AR_SAVE_DIR}/${name} \
            "${ar_global_args[@]}" "${stage3_only[@]}" \
            > ${AR_LOGDIR}/${name}.log 2>&1
        if [ $? -ne 0 ]; then
            echo "ERROR: [${name}] Phase 3 failed."
            return 1
        fi
    else
        echo "===== [${name}] Phase 3: SKIPPED ====="
    fi

    echo "===== [${name}] Done ====="
}

# _eval_run <log_path> <wandb_tag> [train_ar_args...]
_eval_run() {
    local log_path=$1; shift
    local wandb_tag=$1; shift
    "${ACCEL_LAUNCH[@]}" --mixed_precision=bf16 \
        train_ar.py \
        wandb_name=${wandb_tag} \
        "$@" \
        > "${log_path}" 2>&1
    local rc=$?
    [ $rc -ne 0 ] && echo "WARNING: eval run '${wandb_tag}' failed (exit ${rc}), continuing..."
    return 0
}

# _extract_gfid <log_file> <key={gFID|gFID_nocfg|IS}>
_extract_gfid() {
    local log_file=$1
    local key=${2:-gFID}
    [ -f "$log_file" ] || { echo "N/A"; return; }
    local val
    if [ "$key" = "gFID" ]; then
        val=$(grep -P '\bgFID: ' "${log_file}" 2>/dev/null \
            | grep -v 'nocfg' | grep -oP '[0-9]+\.[0-9]+' | head -n 1)
    elif [ "$key" = "IS" ]; then
        val=$(grep -P '\bIS: ' "${log_file}" 2>/dev/null \
            | grep -oP '[0-9]+\.[0-9]+' | tail -n 1)
    else
        val=$(grep -P 'gFID_nocfg: ' "${log_file}" 2>/dev/null \
            | grep -oP '[0-9]+\.[0-9]+' | tail -n 1)
    fi
    echo "${val:-N/A}"
}

# sweep_cfg <tok_preset> <name> [extra_args...]
# Local keys: tokenizer_ckpt_path / ar_ckpt_path / pretoken_dir / extra_configs / ar_stage / cfg_grid / skip_done
# cfg_grid entries: "oc:<cfg>:<pow>" | "sc:<sc>:<vc>:<vp>" | "nt:<t>" | "nt2:<st>:<vt>"
sweep_cfg() {
    local tok=$1; shift
    local tok_configs="${tok%%|*}"
    local name=$1; shift

    # Strip AE-only keys so resume_ckpt_path is interpreted as AR-load here.
    local user_tok_ckpt=""
    local user_ar_ckpt=""
    local user_pretoken_dir=""
    local extra_configs=""
    local ar_stage=2
    local cfg_grid=""
    local skip_done=true
    local extra_args=()
    for arg in "$@"; do
        case "$arg" in
            tokenizer_ckpt_path=*) user_tok_ckpt="${arg#tokenizer_ckpt_path=}" ;;
            ar_ckpt_path=*)        user_ar_ckpt="${arg#ar_ckpt_path=}" ;;
            pretoken_dir=*)        user_pretoken_dir="${arg#pretoken_dir=}" ;;
            extra_configs=*)       extra_configs="${arg#extra_configs=}" ;;
            ar_stage=*)            ar_stage="${arg#ar_stage=}" ;;
            cfg_grid=*)            cfg_grid="${arg#cfg_grid=}" ;;
            skip_done=*)           skip_done="${arg#skip_done=}" ;;
            resume_ckpt_path=*|resume_enc=*|resume_dec=*|resume_gan=*|phases=*) ;;
            *)                     extra_args+=("$arg") ;;
        esac
    done

    if [ -z "${cfg_grid}" ]; then
        echo "ERROR: [${name}] cfg_grid= is required (e.g. cfg_grid=\"sc:0.7:3.75:0.2\")"
        return 1
    fi

    local tok_ckpt="${user_tok_ckpt}"
    if [ -z "$tok_ckpt" ]; then
        tok_ckpt=$(find_best_ckpt ${name})
    fi
    if [ -z "$tok_ckpt" ]; then
        echo "ERROR: [${name}] No tokenizer checkpoint found in ${AE_SAVE_DIR}"
        return 1
    fi

    local pretoken_dir="${user_pretoken_dir}"
    if [ -z "${pretoken_dir}" ]; then
        local resolved
        resolved=$(resolve_pretoken_and_tok "${tok_ckpt}")
        if [ $? -eq 0 ] && [ -n "$resolved" ]; then
            local resolved_pretoken_dir="${resolved%%|*}"
            local resolved_tok_ckpt="${resolved##*|}"
            pretoken_dir="${resolved_pretoken_dir}"
            if [ "${resolved_tok_ckpt}" != "${tok_ckpt}" ]; then
                echo "  [pretoken fallback] tok ckpt: ${tok_ckpt##*/} -> ${resolved_tok_ckpt##*/}"
                tok_ckpt="${resolved_tok_ckpt}"
            fi
        else
            echo "WARNING: [${name}] No pretoken dir found, train_ar.py will infer from ckpt path"
        fi
    fi

    local ar_ckpt=""
    if [ "${ar_stage}" = "1" ]; then
        echo "  [ar_stage=1] Using Phase-1 AR config (AR_JOINT) and weights from tokenizer ckpt"
    else
        ar_ckpt="${user_ar_ckpt}"
        if [ -z "$ar_ckpt" ]; then
            ar_ckpt=$(find_best_ar_ckpt ${name})
        fi
        if [ -z "$ar_ckpt" ]; then
            echo "ERROR: [${name}] No AR checkpoint found in ${AR_SAVE_DIR}"
            return 1
        fi
    fi

    local ar_cfg="$AR"
    [ "${ar_stage}" = "1" ] && ar_cfg="$AR_JOINT"
    local cfg_configs="$BASE,${ar_cfg},$tok_configs,$TRAIN_AR,$TRAIN_EVAL_AR"
    [ -n "$extra_configs" ] && cfg_configs="${cfg_configs},${extra_configs}"

    local log_dir="${EVAL_LOGDIR}/${name}"
    mkdir -p "${log_dir}"

    echo "===== [${name}] CFG eval: ar_stage=${ar_stage} AR=${ar_cfg##*,} ====="
    echo "  tok ckpt     : ${tok_ckpt}"
    if [ "${ar_stage}" = "1" ]; then
        echo "  AR  weights  : from tokenizer ckpt (continuous_training)"
    else
        echo "  AR  ckpt     : ${ar_ckpt}"
    fi
    echo "  pretoken_dir : ${pretoken_dir:-(inferred by train_ar.py)}"
    [ -n "$extra_configs" ] && echo "  extra cfgs: ${extra_configs}"
    [ ${#extra_args[@]} -gt 0 ] && echo "  extra args: ${extra_args[*]}"
    echo "  cfg_grid     : $(echo ${cfg_grid} | wc -w) entries"

    local common_args=(
        --configs=${cfg_configs}
        tokenizer_ckpt_path=${tok_ckpt}
        "${extra_args[@]}"
    )
    if [ "${ar_stage}" = "1" ]; then
        common_args+=(continuous_training=True)
    else
        common_args+=(resume_ckpt_path=${ar_ckpt})
    fi
    [ -n "$pretoken_dir" ] && common_args+=(pretoken_dir=${pretoken_dir})

    local summary_parts=()
    local _skipped=0

    for entry in ${cfg_grid}; do
        local _tag="" _args=()
        case "$entry" in
            oc:*)
                local _rest="${entry#oc:}"
                local _cfg="${_rest%%:*}" _pow="${_rest#*:}"
                _tag="overall-cosine(${_cfg},p=${_pow})"
                _args=(cfg=${_cfg} cfg_schedule=cosine cfg_power=${_pow})
                ;;
            sc:*)
                local _rest="${entry#sc:}"
                IFS=: read -r _scfg _vcfg _vpow <<< "${_rest}"
                _tag="sem=const(${_scfg})_vis=cosine(${_vcfg},p=${_vpow})"
                _args=(semantic_cfg_schedule=constant semantic_cfg_scale=${_scfg}
                       visual_cfg_schedule=cosine visual_cfg_scale=${_vcfg} visual_cfg_power=${_vpow})
                ;;
            nt:*)
                local _temp="${entry#nt:}"
                _tag="nocfg_temp=${_temp}"
                _args=(cfg=0 cfg_schedule=constant temperature=${_temp})
                ;;
            nt2:*)
                local _rest="${entry#nt2:}"
                IFS=: read -r _stemp _vtemp <<< "${_rest}"
                _tag="nocfg_stemp=${_stemp}_vtemp=${_vtemp}"
                _args=(cfg=0 cfg_schedule=constant semantic_temperature=${_stemp} temperature=${_vtemp})
                ;;
            *)
                echo "WARNING: unknown cfg_grid entry '${entry}', skipping"
                continue
                ;;
        esac
        local _log="${log_dir}/${_tag}.log"
        if [ "${skip_done}" = "true" ] && grep -q 'FID_RESULT:' "${_log}" 2>/dev/null; then
            local gfid_val=$(_extract_gfid "${_log}" "gFID")
            summary_parts+=("$(printf '%s\t%s' "${gfid_val}" "${_tag}")")
            echo "===== [${name}] SKIP ${_tag} (gFID=${gfid_val}) ====="
            ((_skipped++)) || true; continue
        fi
        echo "===== [${name}] CFG ${_tag} ====="
        _eval_run "${_log}" "${name}-${_tag}" \
            "${common_args[@]}" "${_args[@]}" \
            nocfg_sample=False do_ar_eval_loader=False
        local gfid_val=$(_extract_gfid "${_log}" "gFID")
        summary_parts+=("$(printf '%s\t%s' "${gfid_val}" "${_tag}")")
    done

    echo ""
    echo "########## [${name}] CFG Eval Summary ##########"
    echo "  tokenizer : ${tok_ckpt}"
    if [ "${ar_stage}" = "1" ]; then
        echo "  AR model  : Phase-1 (from tokenizer ckpt)"
    else
        echo "  AR model  : ${ar_ckpt}"
    fi
    echo "  ar_stage  : ${ar_stage}"
    echo "  total runs: ${#summary_parts[@]}, skipped: ${_skipped}"
    echo ""
    echo "--- Results (sorted by gFID) ---"
    printf '%s\n' "${summary_parts[@]}" | sort -t$'\t' -k1 -n
    echo "################################################"
    echo ""
}
