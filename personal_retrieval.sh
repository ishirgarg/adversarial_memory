#!/usr/bin/env bash
# personal_retrieval.sh
#
# Misleading-persona / persona-retrieval end-to-end pipeline:
#   1. Generate misleading-persona dataset
#   2. Evaluate mem0, simplemem, amem
#   3. Grade all three with analyze_errors.py
#
# Usage:
#   ./personal_retrieval.sh [options]
#
# Skip flags:
#   --skip-generate       Skip dataset generation (use existing CSV)
#   --skip-mem0           Skip mem0 evaluation + analysis
#   --skip-simplemem      Skip simplemem evaluation + analysis
#   --skip-amem           Skip amem evaluation + analysis
#   --skip-evermemos      Skip evermemos evaluation + analysis (default: skipped)
#   --skip-structmem      Skip structmem evaluation + analysis (default: skipped)
#   --skip-analysis       Skip all error analysis steps
#
# Core:
#   --dataset PATH              Path to dataset CSV (default: datasets/custom_persona_retrieval/...)
#   --num-rows N                Rows to generate (default: 100)
#   --seed N                    Random seed for generation (default: 42)
#   --llm-model MODEL           Test-taker LLM model (default: gpt-5-mini)
#   --judge-model MODEL         Judge LLM model (default: gpt-5-mini)
#   --judge-workers N           Parallel judge workers (default: 8)
#   --num-memories N            Retrieval top-k (default: 5)
#   --facts-per-group N         Essays per storage conversation (default: 1)
#   --shared-user-id ID         Shared user-id namespace (default: persona_retrieval_eval_user)
#   --run-id NAME               Unique results subfolder (default: run_<timestamp>).
#                               Outputs land in playground/custom_persona_retrieval/results/<RUN_ID>/<memory>/
#
# mem0:
#   --mem0-llm-provider P       (default: openai)
#   --mem0-llm-model M          (default: gpt-4.1-mini)
#   --mem0-embedding-provider P (default: none)
#   --mem0-embedding-model M    (default: none)
#
# amem:
#   --amem-llm-backend B        (default: openai)
#   --amem-llm-model M          (default: gpt-4.1-mini)
#   --amem-embedding-model M    (default: all-MiniLM-L6-v2)
#   --amem-evo-threshold N      (default: 100)
#
# simplemem:
#   --simplemem-model M                    (default: gpt-4.1-mini)
#   --simplemem-embedding-model M          (default: all-MiniLM-L6-v2)
#   --simplemem-embedding-dimension N      (default: 384)
#   --simplemem-embedding-context-length N (default: 512)
#   --simplemem-db-path PATH               (default: ./lancedb_data)
#   --simplemem-window-size N              (default: 20)
#   --simplemem-overlap-size N             (default: 2)
#   --simplemem-memory-table-name NAME     (default: memory_entries)
#   --simplemem-max-parallel-workers N     (default: 16)
#   --simplemem-max-retrieval-workers N    (default: 8)
#   --simplemem-max-reflection-rounds N    (default: 2)
#
# evermemos (requires EverCore HTTP server, default localhost:1995):
#   --evermemos-base-url URL          (default: http://localhost:1995)
#   --evermemos-llm-provider P        Override boundary+extraction provider via /api/v1/settings
#   --evermemos-llm-model M           Override boundary+extraction model
#   --evermemos-retrieve-method M     keyword|vector|hybrid|agentic (default: hybrid)
#
# structmem (requires vendored LightMem at <repo_root>/LightMem):
#   --structmem-model M               (default: gpt-4.1-mini)
#   --structmem-api-key KEY
#   --structmem-base-url URL
#   --structmem-embedding-model M     (default: sentence-transformers/all-MiniLM-L6-v2)
#   --structmem-qdrant-path PATH      (default: ./structmem_qdrant)
#   --structmem-collection-name NAME  (default: structmem)

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

# Skip flags
SKIP_GENERATE=true
SKIP_MEM0=false
SKIP_SIMPLEMEM=false
SKIP_AMEM=false
SKIP_EVERMEMOS=true
SKIP_STRUCTMEM=true
SKIP_LICOMEMORY=true
SKIP_ANALYSIS=false

# Dataset
DATASET="datasets/custom_persona_retrieval/misleading_persona_dataset.csv"
NUM_ROWS=100
SEED=42

# Core eval
LLM_MODEL="gpt-5-mini"
NUM_MEMORIES=5
FACTS_PER_GROUP=1
SHARED_USER_ID="persona_retrieval_eval_user"
RUN_ID=""

# Judge
JUDGE_MODEL="gpt-5-mini"
JUDGE_WORKERS=8

# mem0
MEM0_LLM_PROVIDER="openai"
MEM0_LLM_MODEL="gpt-4.1-mini"
MEM0_LLM_API_KEY=""
MEM0_LLM_BASE_URL=""
MEM0_EMBEDDING_PROVIDER=""
MEM0_EMBEDDING_MODEL=""

# amem
AMEM_LLM_BACKEND="openai"
AMEM_LLM_MODEL="gpt-4.1-mini"
AMEM_API_KEY=""
AMEM_BASE_URL=""
AMEM_EMBEDDING_MODEL="all-MiniLM-L6-v2"
AMEM_EVO_THRESHOLD=100

# simplemem
SIMPLEMEM_MODEL="gpt-4.1-mini"
SIMPLEMEM_API_KEY=""
SIMPLEMEM_BASE_URL=""
SIMPLEMEM_EMBEDDING_MODEL="all-MiniLM-L6-v2"
SIMPLEMEM_EMBEDDING_DIMENSION=384
SIMPLEMEM_EMBEDDING_CONTEXT_LENGTH=512
SIMPLEMEM_DB_PATH="./lancedb_data"
SIMPLEMEM_WINDOW_SIZE=20
SIMPLEMEM_OVERLAP_SIZE=2
SIMPLEMEM_MEMORY_TABLE_NAME="memory_entries"
SIMPLEMEM_MAX_PARALLEL_WORKERS=16
SIMPLEMEM_MAX_RETRIEVAL_WORKERS=8
SIMPLEMEM_MAX_REFLECTION_ROUNDS=2

# evermemos
EVERMEMOS_BASE_URL="http://localhost:1995"
EVERMEMOS_LLM_PROVIDER=""
EVERMEMOS_LLM_MODEL=""
EVERMEMOS_RETRIEVE_METHOD="hybrid"

# structmem
STRUCTMEM_MODEL="gpt-4.1-mini"
STRUCTMEM_API_KEY=""
STRUCTMEM_BASE_URL=""
STRUCTMEM_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
STRUCTMEM_QDRANT_PATH="./structmem_qdrant"
STRUCTMEM_COLLECTION_NAME="structmem"

# licomemory
LICOMEMORY_MODEL="gpt-4.1-mini"
LICOMEMORY_API_KEY=""
LICOMEMORY_BASE_URL=""

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Skip flags
        --skip-generate)   SKIP_GENERATE=true;  shift ;;
        --skip-mem0)       SKIP_MEM0=true;       shift ;;
        --skip-simplemem)  SKIP_SIMPLEMEM=true;  shift ;;
        --skip-amem)       SKIP_AMEM=true;       shift ;;
        --skip-evermemos)  SKIP_EVERMEMOS=true;  shift ;;
        --skip-structmem)  SKIP_STRUCTMEM=true;  shift ;;
        --skip-licomemory) SKIP_LICOMEMORY=true; shift ;;
        --run-evermemos)   SKIP_EVERMEMOS=false; shift ;;
        --run-structmem)   SKIP_STRUCTMEM=false; shift ;;
        --run-licomemory)  SKIP_LICOMEMORY=false; shift ;;
        --skip-analysis)   SKIP_ANALYSIS=true;   shift ;;

        # Dataset / generation
        --dataset)    DATASET="$2";  shift 2 ;;
        --num-rows)   NUM_ROWS="$2"; shift 2 ;;
        --seed)       SEED="$2";     shift 2 ;;

        # Core eval
        --llm-model)       LLM_MODEL="$2";       shift 2 ;;
        --num-memories)    NUM_MEMORIES="$2";     shift 2 ;;
        --facts-per-group) FACTS_PER_GROUP="$2";  shift 2 ;;
        --shared-user-id)  SHARED_USER_ID="$2";   shift 2 ;;
        --run-id)          RUN_ID="$2";           shift 2 ;;

        # Judge
        --judge-model)    JUDGE_MODEL="$2";    shift 2 ;;
        --judge-workers)  JUDGE_WORKERS="$2";  shift 2 ;;

        # mem0
        --mem0-llm-provider)       MEM0_LLM_PROVIDER="$2";       shift 2 ;;
        --mem0-llm-model)          MEM0_LLM_MODEL="$2";           shift 2 ;;
        --mem0-llm-api-key)        MEM0_LLM_API_KEY="$2";         shift 2 ;;
        --mem0-llm-base-url)       MEM0_LLM_BASE_URL="$2";        shift 2 ;;
        --mem0-embedding-provider) MEM0_EMBEDDING_PROVIDER="$2";  shift 2 ;;
        --mem0-embedding-model)    MEM0_EMBEDDING_MODEL="$2";     shift 2 ;;

        # amem
        --amem-llm-backend)     AMEM_LLM_BACKEND="$2";     shift 2 ;;
        --amem-llm-model)       AMEM_LLM_MODEL="$2";       shift 2 ;;
        --amem-api-key)         AMEM_API_KEY="$2";         shift 2 ;;
        --amem-base-url)        AMEM_BASE_URL="$2";        shift 2 ;;
        --amem-embedding-model) AMEM_EMBEDDING_MODEL="$2"; shift 2 ;;
        --amem-evo-threshold)   AMEM_EVO_THRESHOLD="$2";   shift 2 ;;

        # simplemem
        --simplemem-model)                    SIMPLEMEM_MODEL="$2";                    shift 2 ;;
        --simplemem-api-key)                  SIMPLEMEM_API_KEY="$2";                  shift 2 ;;
        --simplemem-base-url)                 SIMPLEMEM_BASE_URL="$2";                 shift 2 ;;
        --simplemem-embedding-model)          SIMPLEMEM_EMBEDDING_MODEL="$2";          shift 2 ;;
        --simplemem-embedding-dimension)      SIMPLEMEM_EMBEDDING_DIMENSION="$2";      shift 2 ;;
        --simplemem-embedding-context-length) SIMPLEMEM_EMBEDDING_CONTEXT_LENGTH="$2"; shift 2 ;;
        --simplemem-db-path)                  SIMPLEMEM_DB_PATH="$2";                  shift 2 ;;
        --simplemem-window-size)              SIMPLEMEM_WINDOW_SIZE="$2";              shift 2 ;;
        --simplemem-overlap-size)             SIMPLEMEM_OVERLAP_SIZE="$2";             shift 2 ;;
        --simplemem-memory-table-name)        SIMPLEMEM_MEMORY_TABLE_NAME="$2";        shift 2 ;;
        --simplemem-max-parallel-workers)     SIMPLEMEM_MAX_PARALLEL_WORKERS="$2";     shift 2 ;;
        --simplemem-max-retrieval-workers)    SIMPLEMEM_MAX_RETRIEVAL_WORKERS="$2";    shift 2 ;;
        --simplemem-max-reflection-rounds)    SIMPLEMEM_MAX_REFLECTION_ROUNDS="$2";    shift 2 ;;

        # evermemos
        --evermemos-base-url)         EVERMEMOS_BASE_URL="$2";        shift 2 ;;
        --evermemos-llm-provider)     EVERMEMOS_LLM_PROVIDER="$2";    shift 2 ;;
        --evermemos-llm-model)        EVERMEMOS_LLM_MODEL="$2";       shift 2 ;;
        --evermemos-retrieve-method)  EVERMEMOS_RETRIEVE_METHOD="$2"; shift 2 ;;

        # structmem
        --structmem-model)            STRUCTMEM_MODEL="$2";            shift 2 ;;
        --structmem-api-key)          STRUCTMEM_API_KEY="$2";          shift 2 ;;
        --structmem-base-url)         STRUCTMEM_BASE_URL="$2";         shift 2 ;;
        --structmem-embedding-model)  STRUCTMEM_EMBEDDING_MODEL="$2";  shift 2 ;;
        --structmem-qdrant-path)      STRUCTMEM_QDRANT_PATH="$2";      shift 2 ;;
        --structmem-collection-name)  STRUCTMEM_COLLECTION_NAME="$2";  shift 2 ;;

        # licomemory
        --licomemory-model)           LICOMEMORY_MODEL="$2";           shift 2 ;;
        --licomemory-api-key)         LICOMEMORY_API_KEY="$2";         shift 2 ;;
        --licomemory-base-url)        LICOMEMORY_BASE_URL="$2";        shift 2 ;;

        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$RUN_ID" ]; then
    RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
fi
RESULTS_BASE="playground/custom_persona_retrieval/results/$RUN_ID"
mkdir -p "$RESULTS_BASE"

MEM0_EMBEDDING_ARGS=""
if [ -n "$MEM0_EMBEDDING_PROVIDER" ]; then
    MEM0_EMBEDDING_ARGS="--mem0-embedding-provider $MEM0_EMBEDDING_PROVIDER"
fi
if [ -n "$MEM0_EMBEDDING_MODEL" ]; then
    MEM0_EMBEDDING_ARGS="$MEM0_EMBEDDING_ARGS --mem0-embedding-model $MEM0_EMBEDDING_MODEL"
fi

MEM0_PROXY_ARGS=""
if [ -n "$MEM0_LLM_API_KEY" ]; then
    MEM0_PROXY_ARGS="--mem0-llm-api-key $MEM0_LLM_API_KEY"
fi
if [ -n "$MEM0_LLM_BASE_URL" ]; then
    MEM0_PROXY_ARGS="$MEM0_PROXY_ARGS --mem0-llm-base-url $MEM0_LLM_BASE_URL"
fi

AMEM_PROXY_ARGS=""
if [ -n "$AMEM_API_KEY" ]; then
    AMEM_PROXY_ARGS="--amem-api-key $AMEM_API_KEY"
fi
if [ -n "$AMEM_BASE_URL" ]; then
    AMEM_PROXY_ARGS="$AMEM_PROXY_ARGS --amem-base-url $AMEM_BASE_URL"
fi

SIMPLEMEM_PROXY_ARGS=""
if [ -n "$SIMPLEMEM_API_KEY" ]; then
    SIMPLEMEM_PROXY_ARGS="--simplemem-api-key $SIMPLEMEM_API_KEY"
fi
if [ -n "$SIMPLEMEM_BASE_URL" ]; then
    SIMPLEMEM_PROXY_ARGS="$SIMPLEMEM_PROXY_ARGS --simplemem-base-url $SIMPLEMEM_BASE_URL"
fi

EVERMEMOS_LLM_ARGS=""
if [ -n "$EVERMEMOS_LLM_PROVIDER" ]; then
    EVERMEMOS_LLM_ARGS="--evermemos-llm-provider $EVERMEMOS_LLM_PROVIDER"
fi
if [ -n "$EVERMEMOS_LLM_MODEL" ]; then
    EVERMEMOS_LLM_ARGS="$EVERMEMOS_LLM_ARGS --evermemos-llm-model $EVERMEMOS_LLM_MODEL"
fi

STRUCTMEM_PROXY_ARGS=""
if [ -n "$STRUCTMEM_API_KEY" ]; then
    STRUCTMEM_PROXY_ARGS="--structmem-api-key $STRUCTMEM_API_KEY"
fi
if [ -n "$STRUCTMEM_BASE_URL" ]; then
    STRUCTMEM_PROXY_ARGS="$STRUCTMEM_PROXY_ARGS --structmem-base-url $STRUCTMEM_BASE_URL"
fi

LICOMEMORY_PROXY_ARGS=""
if [ -n "$LICOMEMORY_API_KEY" ]; then
    LICOMEMORY_PROXY_ARGS="--licomemory-api-key $LICOMEMORY_API_KEY"
fi
if [ -n "$LICOMEMORY_BASE_URL" ]; then
    LICOMEMORY_PROXY_ARGS="$LICOMEMORY_PROXY_ARGS --licomemory-base-url $LICOMEMORY_BASE_URL"
fi

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY=$(git diff --quiet 2>/dev/null && echo "clean" || echo "dirty")

echo "============================================================"
echo "  Persona Retrieval Pipeline"
echo "============================================================"
echo ""
echo "  --- Git ---"
echo "  Commit:             $GIT_COMMIT ($GIT_DIRTY)"
echo ""
echo "  --- Pipeline steps ---"
echo "  Run generate:       $([ "$SKIP_GENERATE" = false ] && echo yes || echo SKIPPED)"
echo "  Run mem0:           $([ "$SKIP_MEM0"     = false ] && echo yes || echo SKIPPED)"
echo "  Run simplemem:      $([ "$SKIP_SIMPLEMEM" = false ] && echo yes || echo SKIPPED)"
echo "  Run amem:           $([ "$SKIP_AMEM"     = false ] && echo yes || echo SKIPPED)"
echo "  Run evermemos:      $([ "$SKIP_EVERMEMOS" = false ] && echo yes || echo SKIPPED)"
echo "  Run structmem:      $([ "$SKIP_STRUCTMEM" = false ] && echo yes || echo SKIPPED)"
echo "  Run analysis:       $([ "$SKIP_ANALYSIS" = false ] && echo yes || echo SKIPPED)"
echo ""
echo "  --- Dataset generation ---"
echo "  Output CSV:         $DATASET"
echo "  Num rows:           $NUM_ROWS"
echo "  Batch size:         10"
echo "  Seed:               $SEED"
echo "  Dedup threshold:    0.7 (MinHash LSH, essay field)"
echo ""
echo "  --- Evaluation (shared) ---"
echo "  LLM model:          $LLM_MODEL"
echo "  Num memories (top-k): $NUM_MEMORIES"
echo "  Facts per group:    $FACTS_PER_GROUP"
echo "  Shared user ID:     $SHARED_USER_ID"
echo "  Run ID (output dir): $RESULTS_BASE"
echo ""
echo "  --- Judge ---"
echo "  Model:              $JUDGE_MODEL"
echo "  Workers:            $JUDGE_WORKERS"
echo ""
echo "  --- mem0 ---"
echo "  LLM provider:       $MEM0_LLM_PROVIDER"
echo "  LLM model:          $MEM0_LLM_MODEL"
echo "  Embedding provider: ${MEM0_EMBEDDING_PROVIDER:-(mem0 default)}"
echo "  Embedding model:    ${MEM0_EMBEDDING_MODEL:-(mem0 default)}"
echo ""
echo "  --- A-MEM ---"
echo "  LLM backend:        $AMEM_LLM_BACKEND"
echo "  LLM model:          $AMEM_LLM_MODEL"
echo "  Embedding model:    $AMEM_EMBEDDING_MODEL"
echo "  Evo threshold:      $AMEM_EVO_THRESHOLD"
echo ""
echo "  --- SimpleMem ---"
echo "  LLM model:          $SIMPLEMEM_MODEL"
echo "  Embedding model:    $SIMPLEMEM_EMBEDDING_MODEL"
echo "  Embedding dim:      $SIMPLEMEM_EMBEDDING_DIMENSION"
echo "  Embedding ctx len:  $SIMPLEMEM_EMBEDDING_CONTEXT_LENGTH"
echo "  DB path:            $SIMPLEMEM_DB_PATH"
echo "  Window size:        $SIMPLEMEM_WINDOW_SIZE"
echo "  Overlap size:       $SIMPLEMEM_OVERLAP_SIZE"
echo "  Memory table name:  $SIMPLEMEM_MEMORY_TABLE_NAME"
echo "  Max parallel workers: $SIMPLEMEM_MAX_PARALLEL_WORKERS"
echo "  Max retrieval workers: $SIMPLEMEM_MAX_RETRIEVAL_WORKERS"
echo "  Max reflection rounds: $SIMPLEMEM_MAX_REFLECTION_ROUNDS"
echo "============================================================"

# --------------------------------------------------------------------------
# Helper: run evaluate + analyze for one memory system
# --------------------------------------------------------------------------
run_memory_system() {
    local MEMORY="$1"
    local LABEL="$2"
    local EXTRA_ARGS="${3:-}"

    echo ""
    echo ">>> Evaluating on $LABEL..."
    # shellcheck disable=SC2086
    uv run python playground/custom_persona_retrieval/evaluate_persona_retrieval.py \
        --dataset "$DATASET" \
        --memory "$MEMORY" \
        --output-dir "$RESULTS_BASE" \
        --run-dir "$RESULTS_BASE" \
        --llm-model "$LLM_MODEL" \
        --num-memories "$NUM_MEMORIES" \
        --facts-per-group "$FACTS_PER_GROUP" \
        --shared-user-id "$SHARED_USER_ID" \
        --seed "$SEED" \
        $EXTRA_ARGS

    local GRADED_TRACES
    GRADED_TRACES=$(ls -t "$RESULTS_BASE/$MEMORY/graded_traces_"*.json 2>/dev/null \
                   | head -1 || echo "")

    if [ -z "$GRADED_TRACES" ]; then
        echo ">>> WARNING: no graded_traces file found for $LABEL, skipping analysis."
        return
    fi
    echo ">>> $LABEL graded traces: $GRADED_TRACES"

    if [ "$SKIP_ANALYSIS" = false ]; then
        echo ">>> Analyzing $LABEL traces..."
        uv run python playground/custom_persona_retrieval/analyze_errors.py \
            --traces "$GRADED_TRACES" \
            --model "$JUDGE_MODEL" \
            --workers "$JUDGE_WORKERS"
    fi
}

# --------------------------------------------------------------------------
# Step 1: Generate dataset
# --------------------------------------------------------------------------
if [ "$SKIP_GENERATE" = false ]; then
    echo ""
    echo ">>> Generating dataset ($NUM_ROWS rows)..."
    uv run python dataset_utils/custom_persona_retrieval/generate_misleading_persona_dataset.py \
        --num-rows "$NUM_ROWS" \
        --batch-size 5 \
        --seed "$SEED" \
        --output-csv "$DATASET"
    echo ">>> Dataset saved to: $DATASET"
else
    echo ""
    echo ">>> Skipping dataset generation. Using: $DATASET"
fi

# --------------------------------------------------------------------------
# Step 2: mem0
# --------------------------------------------------------------------------
if [ "$SKIP_MEM0" = false ]; then
    run_memory_system "mem0" "mem0" \
        "--mem0-llm-provider $MEM0_LLM_PROVIDER \
         --mem0-llm-model $MEM0_LLM_MODEL \
         $MEM0_EMBEDDING_ARGS \
         $MEM0_PROXY_ARGS"
else
    echo ""
    echo ">>> Skipping mem0."
fi

# --------------------------------------------------------------------------
# Step 3: simplemem
# --------------------------------------------------------------------------
if [ "$SKIP_SIMPLEMEM" = false ]; then
    run_memory_system "simplemem" "SimpleMem" \
        "--simplemem-model $SIMPLEMEM_MODEL \
         --simplemem-embedding-model $SIMPLEMEM_EMBEDDING_MODEL \
         --simplemem-embedding-dimension $SIMPLEMEM_EMBEDDING_DIMENSION \
         --simplemem-embedding-context-length $SIMPLEMEM_EMBEDDING_CONTEXT_LENGTH \
         --simplemem-db-path $SIMPLEMEM_DB_PATH \
         --simplemem-window-size $SIMPLEMEM_WINDOW_SIZE \
         --simplemem-overlap-size $SIMPLEMEM_OVERLAP_SIZE \
         --simplemem-memory-table-name $SIMPLEMEM_MEMORY_TABLE_NAME \
         --simplemem-max-parallel-workers $SIMPLEMEM_MAX_PARALLEL_WORKERS \
         --simplemem-max-retrieval-workers $SIMPLEMEM_MAX_RETRIEVAL_WORKERS \
         --simplemem-max-reflection-rounds $SIMPLEMEM_MAX_REFLECTION_ROUNDS \
         $SIMPLEMEM_PROXY_ARGS"
else
    echo ""
    echo ">>> Skipping simplemem."
fi

# --------------------------------------------------------------------------
# Step 4: amem
# --------------------------------------------------------------------------
if [ "$SKIP_AMEM" = false ]; then
    run_memory_system "amem" "A-MEM" \
        "--amem-llm-backend $AMEM_LLM_BACKEND \
         --amem-llm-model $AMEM_LLM_MODEL \
         --amem-embedding-model $AMEM_EMBEDDING_MODEL \
         --amem-evo-threshold $AMEM_EVO_THRESHOLD \
         $AMEM_PROXY_ARGS"
else
    echo ""
    echo ">>> Skipping amem."
fi

# --------------------------------------------------------------------------
# Step 5: evermemos
# --------------------------------------------------------------------------
if [ "$SKIP_EVERMEMOS" = false ]; then
    run_memory_system "evermemos" "EverMemOS" \
        "--evermemos-base-url $EVERMEMOS_BASE_URL \
         --evermemos-retrieve-method $EVERMEMOS_RETRIEVE_METHOD \
         $EVERMEMOS_LLM_ARGS"
else
    echo ""
    echo ">>> Skipping evermemos."
fi

# --------------------------------------------------------------------------
# Step 6: structmem
# --------------------------------------------------------------------------
if [ "$SKIP_STRUCTMEM" = false ]; then
    run_memory_system "structmem" "StructMem" \
        "--structmem-model $STRUCTMEM_MODEL \
         --structmem-embedding-model $STRUCTMEM_EMBEDDING_MODEL \
         --structmem-qdrant-path $STRUCTMEM_QDRANT_PATH \
         --structmem-collection-name $STRUCTMEM_COLLECTION_NAME \
         $STRUCTMEM_PROXY_ARGS"
else
    echo ""
    echo ">>> Skipping structmem."
fi

# --------------------------------------------------------------------------
# Step 7: licomemory
# --------------------------------------------------------------------------
if [ "$SKIP_LICOMEMORY" = false ]; then
    run_memory_system "licomemory" "LiCoMemory" \
        "--licomemory-model $LICOMEMORY_MODEL \
         $LICOMEMORY_PROXY_ARGS"
else
    echo ""
    echo ">>> Skipping licomemory."
fi

echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "  Results in: $RESULTS_BASE/"
echo "============================================================"
