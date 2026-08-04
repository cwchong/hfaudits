#!/usr/bin/env bash
# =============================================================================
#  hf2docker.sh — Hugging Face safetensors -> Docker images, on a small disk,
#                 with a retained provenance/scan audit record
# =============================================================================
#
#  Layout produced
#  ---------------
#    image name : <namespace>/<image-name>:<model-slug>-shard<N>
#                 e.g. mlbakery:qwen3-coder-next-fp8-shard1
#    payload    : /models/<MODEL_NAME>/...        (upstream casing preserved)
#                 e.g. /models/Qwen3-Coder-Next-FP8/model-00001-of-00009.safetensors
#
#  The images are built on busybox by default, pinned by digest, with CMD set to
#  a shell and WORKDIR set to the model directory, so you can inspect one with:
#      docker run -it --rm <image>            # lands in /models/<MODEL_NAME>
#  Use '-b debian:stable-slim --shell /bin/bash' for real bash, or '-b scratch'
#  for a pure data image with no shell (only usable via COPY --from / image volumes).
#
#  Grouping
#  --------
#    Small shards can be packed together so each image is a useful size:
#      --per-image N          N safetensors per image (default 1; 0 = no count cap)
#      --max-image-size SIZE  close a group early if it would exceed SIZE
#    e.g. --per-image 4 --max-image-size 6GB  =>  up to 4 shards, never over 6 GiB.
#    Group 1 additionally carries the repo metadata files and the audit bundle.
#
#  Per repo, the script:
#    1. Resolves the revision to an immutable commit SHA (so every shard comes
#       from one snapshot even if `main` moves mid-run).
#    2. Captures and retains the AUDIT BUNDLE for that snapshot:
#         - raw upstream JSON (tree, paths-info, model-info) verbatim
#         - normalised per-file record: size, git blob oid, LFS sha256, Xet
#           hash, last commit, and every scanner verdict the Hub reports
#         - SHA256SUMS, HTTP receipts, locally computed sha256 of every byte
#         - resulting image tags and pushed manifest digests
#    3. Applies the scan policy to everything it intends to ship (before any
#       bytes move), then for each GROUP, SEQUENTIALLY:
#         a. verifies free space for the whole group + its image layer + slack,
#            pruning Docker build cache / dangling images first if needed;
#         b. downloads the group's shards into a fresh build context (resumable);
#         c. adds metadata files + audit bundle if this is group 1;
#         d. builds one image, pushes it, records the digest;
#         e. deletes the files, removes the local image, reclaims space.
#
#  Peak disk use is therefore bounded by roughly 2.5x the LARGEST GROUP, not by
#  the size of the repo. With 6 GiB groups budget ~17 GiB free.
#  The audit bundle is text-only and is never touched by the reclaim step.
#
#  Requirements: bash >= 4, curl, jq, docker (logged in), awk, sed, sort, split,
#                sha256sum or shasum
#  Optional env: HF_TOKEN (gated/private repos), DOCKER_NS, HF_ENDPOINT
#
#  Examples:
#    ./hf2docker.sh -n myns --per-image 4 --max-image-size 6GB \
#                   Qwen/Qwen3-Coder-Next-FP8
#    ./hf2docker.sh -n myns --dry-run --max-image-size 6GB -f repos.txt
#
# =============================================================================
set -Eeuo pipefail

PROG=${0##*/}
VERSION=3.0.0

# ------------------------------------------------------------------ defaults --
WORKDIR=${WORKDIR:-${PWD}/hf2docker-work}
AUDIT_DIR=${AUDIT_DIR:-}              # defaults to $WORKDIR/audit
DOCKER_NS=${DOCKER_NS:-}              # Docker Hub namespace (user or org)
IMAGE_NAME=${IMAGE_NAME:-mlbakery}    # image repo; a value containing "/" is used verbatim
TAG_PREFIX=${TAG_PREFIX:-}            # optional prefix on every tag
SHARD_WORD=${SHARD_WORD:-shard}       # tag becomes <model>-<SHARD_WORD><N>
FIRST_ALIAS=${FIRST_ALIAS:-}          # extra tag <model>-TAG on group 1; off by default
MODEL_NAME=${MODEL_NAME:-}            # override the in-image folder name
MODELS_ROOT=${MODELS_ROOT:-/models}   # parent dir inside the image
REVISION=${REVISION:-main}
BASE_IMAGE=${BASE_IMAGE:-debian:12-slim}  # needs a shell so you can exec into it
SHELL_PATH=${SHELL_PATH:-/bin/bash}     # CMD for the image; /bin/bash on a debian base
PLATFORM=${PLATFORM:-linux/amd64}
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

PER_IMAGE=${PER_IMAGE:-1}             # safetensors per image; 0 = no count cap
MAX_IMAGE_SIZE=${MAX_IMAGE_SIZE:-}    # e.g. 6GB; empty = no size cap
MAX_IMAGE_BYTES=0

SPACE_PCT=${SPACE_PCT:-250}           # required free space as % of group size
RESERVE_MB=${RESERVE_MB:-2048}        # absolute headroom kept free, on top of that
META_MAX_MB=${META_MAX_MB:-64}        # skip "meta" files larger than this
PATHS_CHUNK=${PATHS_CHUNK:-500}       # paths per paths-info POST
RETRIES=${RETRIES:-5}

DRY_RUN=0
KEEP=0
SKIP_EXISTING=1
PRUNE=1
USE_BUILDX=0
LOCAL_SHA=1                           # hash every downloaded file for the record
EMBED_AUDIT=1                         # ship the audit bundle inside the group-1 image
REQUIRE_SAFE=0                        # refuse any scanner verdict that is not "safe"
REQUIRE_SCANNED=0                     # additionally refuse files with no scan result
REPOS=()

# Extensions treated as weights (never bundled as "meta" files).
WEIGHT_GLOBS='*.safetensors *.bin *.pt *.pth *.ckpt *.gguf *.ggml *.msgpack *.h5 *.hdf5 *.onnx *.onnx_data *.tflite *.npz *.pkl'

# ------------------------------------------------------------------- logging --
ts()   { date -u +%Y-%m-%dT%H:%M:%SZ; }
log()  { printf '%s [ info] %s\n' "$(ts)" "$*" >&2; }
warn() { printf '%s [ warn] %s\n' "$(ts)" "$*" >&2; }
err()  { printf '%s [error] %s\n' "$(ts)" "$*" >&2; }
die()  { err "$*"; exit 1; }

human() {
  awk -v b="${1:-0}" 'BEGIN{
    split("B KiB MiB GiB TiB PiB",u," "); i=1
    while (b>=1024 && i<6) { b/=1024; i++ }
    printf (i==1 ? "%d %s" : "%.2f %s"), b, u[i]
  }'
}

# to_bytes 6GB|6G|6GiB|6442450944 -> bytes. K/M/G/T are binary multiples.
to_bytes() {
  local s=${1// /} n u mult=1
  [[ $s =~ ^([0-9]+([.][0-9]+)?)([a-zA-Z]*)$ ]] || die "cannot parse size: $1"
  n=${BASH_REMATCH[1]}
  u=$(printf '%s' "${BASH_REMATCH[3]}" | tr '[:upper:]' '[:lower:]')
  case $u in
    ''|b)     mult=1 ;;
    k|kb|kib) mult=1024 ;;
    m|mb|mib) mult=$((1024**2)) ;;
    g|gb|gib) mult=$((1024**3)) ;;
    t|tb|tib) mult=$((1024**4)) ;;
    *) die "unknown size unit '$u' in '$1'" ;;
  esac
  awk -v n="$n" -v m="$mult" 'BEGIN{ printf "%d", n*m }'
}

usage() {
  cat <<EOF
$PROG $VERSION — pack Hugging Face safetensors into Docker images on a small
disk, retaining a full provenance/scan audit record.

Usage: $PROG -n <dockerhub-namespace> [options] <repo> [<repo> ...]

Naming and layout:
  image   <namespace>/<image-name>:<model-slug>-shard<N>
  payload <models-root>/<MODEL_NAME>/...      (upstream casing preserved)
  e.g.    mlbakery:qwen3-coder-next-fp8-shard1
          /models/Qwen3-Coder-Next-FP8/model-00001-of-00009.safetensors

Grouping:
      --per-image N        safetensors per image      (default: $PER_IMAGE; 0 = no count cap)
      --max-image-size SZ  cap group payload, e.g. 6GB (default: none)

Options:
  -n, --namespace NS     Docker Hub user/org to push to        (env DOCKER_NS)
  -i, --image-name NAME  Image repo name                        (default: $IMAGE_NAME)
                         Contains "/"? used verbatim, e.g. ghcr.io/org/mlbakery
  -f, --repos-file FILE  Read repo ids from FILE (one per line, # comments ok)
  -r, --revision REV     Branch/tag/commit to pull             (default: $REVISION)
  -w, --workdir DIR      Scratch directory                     (default: $WORKDIR)
  -a, --audit-dir DIR    Where audit bundles are retained      (default: <workdir>/audit)
  -b, --base IMAGE       Base image for the payload images     (default: $BASE_IMAGE)
                         Use 'scratch' for a pure data image with no shell.
      --shell PATH       Image CMD, so 'docker run -it' drops you in a shell
                         (default: $SHELL_PATH; use /bin/bash on a debian base)
      --models-root PATH Parent dir inside the image           (default: $MODELS_ROOT)
      --model-name NAME  Override the in-image folder name (single repo only)
  -p, --platform P       Image platform                        (default: $PLATFORM)
  -t, --tag-prefix S     Prefix added to every tag
      --shard-word W     Word used in the tag                  (default: $SHARD_WORD)
      --first-alias TAG  Extra tag <model>-TAG on group 1      (default: none)
      --space-pct N      Free space needed, % of group size    (default: $SPACE_PCT)
      --reserve-mb N     Extra headroom kept free, MiB         (default: $RESERVE_MB)
      --meta-max-mb N    Max size of a bundled meta file, MiB  (default: $META_MAX_MB)
      --require-safe     Refuse any file whose scan status is not "safe"
      --require-scanned  Also refuse files with no scan result yet
      --no-local-sha     Skip local sha256 (faster; weakens the audit record)
      --no-embed-audit   Do not copy the audit bundle into the group-1 image
      --buildx           Build with buildx straight to registry (no local image)
      --no-skip-existing Rebuild tags that already exist in the registry
      --no-prune         Never prune Docker build cache / dangling images
      --keep             Keep downloaded files (disables the disk guarantee)
      --dry-run          Collect the audit bundle and show the plan; no downloads
  -h, --help             This help

Audit bundle layout (retained; not pruned):
  <audit-dir>/inventory.tsv        append-only: every file -> image -> digest
  <audit-dir>/<org>__<name>/<commit>/
      tree.json         raw GET  /api/models/<repo>/tree/<rev>?recursive=1&expand=1
      paths-info.json   raw POST /api/models/<repo>/paths-info/<rev>  (chunked)
      model-info.json   raw GET  /api/models/<repo>?securityStatus=true
      files.json        normalised per-file record (sizes, shas, scanner verdicts)
      files.jsonl       same, one object per line, for log/SIEM ingestion
      files.tsv         flat view for grepping / spreadsheets
      SHA256SUMS        coreutils format, usable with 'sha256sum -c'
      receipts.jsonl    per-download HTTP receipt + locally computed sha256
      groups.tsv        how files were packed into images, per run
      images.tsv        image tag -> pushed manifest digest
      audit.json        run manifest (tool version, who, when, endpoint)
EOF
}

# ------------------------------------------------------------- arg parsing --
# read_repos_file <path> -> appends repo ids to REPOS
# `read` returns non-zero on a final line with no trailing newline, so the plain
# `while read` idiom silently drops it; the `|| [[ -n $line ]]` clause catches
# that case. `line` must then be cleared at the end of the body or the same
# non-empty value re-triggers the clause and the loop never terminates.
read_repos_file() {
  local f=$1 line n=0
  [[ -r $f ]] || die "cannot read $f"
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%%#*}                  # strip comments
    line=${line//[[:space:]]/}        # strip all whitespace, incl. CR from CRLF
    if [[ -n $line ]]; then REPOS+=("$line"); n=$(( n + 1 )); fi
    line=""
  done < "$f"
  (( n )) || die "no repositories found in $f"
  log "read $n repo(s) from $f"
}

while (( $# )); do
  # Every option below consumes a value; catch a missing one or a repeated flag
  # (e.g. "--per-image --per-image 3") here, with a message that says so.
  case $1 in
    -n|--namespace|-i|--image-name|-f|--repos-file|-r|--revision|-w|--workdir|\
    -a|--audit-dir|-b|--base|--models-root|--model-name|--per-image|\
    --max-image-size|-p|--platform|-t|--tag-prefix|--shard-word|--first-alias|\
    --shell|--space-pct|--reserve-mb|--meta-max-mb|--paths-chunk)
      (( $# >= 2 )) || die "option $1 requires a value"
      if [[ $2 == -?* ]]; then
        die "option $1 requires a value but got '$2' — repeated flag or missing argument?"
      fi
      ;;
  esac
  case $1 in
    -n|--namespace)     DOCKER_NS=$2; shift 2 ;;
    -i|--image-name)    IMAGE_NAME=$2; shift 2 ;;
    -f|--repos-file)    read_repos_file "$2"; shift 2 ;;
    -r|--revision)      REVISION=$2; shift 2 ;;
    -w|--workdir)       WORKDIR=$2; shift 2 ;;
    -a|--audit-dir)     AUDIT_DIR=$2; shift 2 ;;
    -b|--base)          BASE_IMAGE=$2; shift 2 ;;
    --shell)            SHELL_PATH=$2; shift 2 ;;
    --models-root)      MODELS_ROOT=${2%/}; shift 2 ;;
    --model-name)       MODEL_NAME=$2; shift 2 ;;
    --per-image)        PER_IMAGE=$2; shift 2 ;;
    --max-image-size)   MAX_IMAGE_SIZE=$2; shift 2 ;;
    -p|--platform)      PLATFORM=$2; shift 2 ;;
    -t|--tag-prefix)    TAG_PREFIX=$2; shift 2 ;;
    --shard-word)       SHARD_WORD=$2; shift 2 ;;
    --first-alias)      FIRST_ALIAS=$2; shift 2 ;;
    --space-pct)        SPACE_PCT=$2; shift 2 ;;
    --reserve-mb)       RESERVE_MB=$2; shift 2 ;;
    --meta-max-mb)      META_MAX_MB=$2; shift 2 ;;
    --paths-chunk)      PATHS_CHUNK=$2; shift 2 ;;
    --require-safe)     REQUIRE_SAFE=1; shift ;;
    --require-scanned)  REQUIRE_SAFE=1; REQUIRE_SCANNED=1; shift ;;
    --no-local-sha)     LOCAL_SHA=0; shift ;;
    --no-embed-audit)   EMBED_AUDIT=0; shift ;;
    --buildx)           USE_BUILDX=1; shift ;;
    --verify-sha)       LOCAL_SHA=1; shift ;;          # back-compat: now the default
    --no-skip-existing) SKIP_EXISTING=0; shift ;;
    --no-prune)         PRUNE=0; shift ;;
    --keep)             KEEP=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    -V|--version)       echo "$PROG $VERSION"; exit 0 ;;
    --)                 shift; while (( $# )); do REPOS+=("$1"); shift; done ;;
    -*)                 die "unknown option: $1 (try --help)" ;;
    *)                  REPOS+=("$1"); shift ;;
  esac
done

# -------------------------------------------------------------- preflight --
(( BASH_VERSINFO[0] >= 4 )) || die "bash >= 4 required (found $BASH_VERSION)"
for c in curl jq docker awk sed sort split; do
  command -v "$c" >/dev/null || die "missing required command: $c"
done
SHA_CMD=()
if   command -v sha256sum >/dev/null; then SHA_CMD=(sha256sum)
elif command -v shasum    >/dev/null; then SHA_CMD=(shasum -a 256)
elif (( LOCAL_SHA )); then warn "no sha256 tool found; disabling local hashing"; LOCAL_SHA=0
fi
(( ${#REPOS[@]} )) || { usage; die "no repositories given"; }
check_int() {  # check_int <flag> <value>
  [[ $2 =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer, got '$2'"
}
check_int --per-image   "$PER_IMAGE"
check_int --space-pct   "$SPACE_PCT"
check_int --reserve-mb  "$RESERVE_MB"
check_int --meta-max-mb "$META_MAX_MB"
check_int --paths-chunk "$PATHS_CHUNK"
(( SPACE_PCT >= 100 )) || die "--space-pct below 100 cannot fit the payload, let alone the image layer"
(( PATHS_CHUNK >= 1 )) || die "--paths-chunk must be at least 1"
[[ -n $MAX_IMAGE_SIZE ]] && MAX_IMAGE_BYTES=$(to_bytes "$MAX_IMAGE_SIZE")
if (( PER_IMAGE == 0 && MAX_IMAGE_BYTES == 0 )); then
  warn "--per-image 0 with no --max-image-size: every shard goes in ONE image;"
  warn "this defeats the bounded-disk design. Consider --max-image-size 6GB."
fi
[[ -n $MODEL_NAME && ${#REPOS[@]} -gt 1 ]] && die "--model-name only makes sense with a single repo"

IMAGE_REPO=$IMAGE_NAME
if [[ $IMAGE_NAME != */* ]]; then
  [[ -n $DOCKER_NS ]] || die "set a namespace with -n, or give -i a full path (e.g. ghcr.io/org/mlbakery)"
  IMAGE_REPO="$(printf '%s' "$DOCKER_NS" | tr '[:upper:]' '[:lower:]')/$IMAGE_NAME"
fi
docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon"
if ! grep -q '"auths"[[:space:]]*:[[:space:]]*{[[:space:]]*"' \
     "${DOCKER_CONFIG:-$HOME/.docker}/config.json" 2>/dev/null; then
  warn "no Docker credentials found; run 'docker login' or pushes will fail"
fi

# ---- base image: pull once, pin by digest, confirm the shell actually exists --
# Pinning means every group in the run shares one identical base even if the tag
# moves mid-run, and the digest goes into the audit record and the image labels.
BASE_REF=$BASE_IMAGE
BASE_DIGEST=""
HAS_SHELL=0
if [[ $BASE_IMAGE == scratch ]]; then
  warn "base is 'scratch': the images will contain no shell, so you cannot"
  warn "'docker run' or 'exec' into them. Drop -b scratch to get one."
else
  log "resolving base image $BASE_IMAGE"
  docker pull --platform "$PLATFORM" "$BASE_IMAGE" >/dev/null 2>&1 \
    || warn "could not pull $BASE_IMAGE; using any locally cached copy"
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 \
    || die "base image $BASE_IMAGE is not available locally and could not be pulled"

  BASE_DIGEST=$(docker image inspect --format \
      '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "$BASE_IMAGE" 2>/dev/null \
    | awk -F@ 'NF>1 {print $2}') || true
  if [[ -n $BASE_DIGEST ]]; then
    base_repo=$BASE_IMAGE
    [[ ${BASE_IMAGE##*/} == *:* ]] && base_repo=${BASE_IMAGE%:*}
    BASE_REF="${base_repo}@${BASE_DIGEST}"
    log "base pinned to $BASE_REF"
  else
    warn "no registry digest for $BASE_IMAGE (local-only image?); using the tag as-is"
  fi

  # Verify the shell we are about to declare as CMD is really in there.
  if docker run --rm --platform "$PLATFORM" --entrypoint "$SHELL_PATH" \
       "$BASE_REF" -c 'exit 0' >/dev/null 2>&1; then
    HAS_SHELL=1
    log "shell $SHELL_PATH present in base image"
  else
    warn "$SHELL_PATH not usable in $BASE_IMAGE — the images will have no CMD."
    warn "For real bash try: -b debian:stable-slim --shell /bin/bash"
  fi
fi

mkdir -p "$WORKDIR"
AUDIT_DIR=${AUDIT_DIR:-$WORKDIR/audit}
mkdir -p "$AUDIT_DIR"
CTX="$WORKDIR/ctx"
BUILD_META="$WORKDIR/build-meta.json"
INVENTORY="$AUDIT_DIR/inventory.tsv"
INV_HEADER=$'run\trepo\tcommit\tmodel\tfile\tbytes\tsha256\timage\tdigest\tgroup\tmeta\tscan_status'
if [[ -f $INVENTORY ]]; then
  if [[ $(head -n1 "$INVENTORY") != "$INV_HEADER" ]]; then
    mv "$INVENTORY" "${INVENTORY%.tsv}-superseded-$(date -u +%Y%m%dT%H%M%SZ).tsv"
    printf '%s\n' "$INV_HEADER" > "$INVENTORY"
  fi
else
  printf '%s\n' "$INV_HEADER" > "$INVENTORY"
fi
DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || true)
RUN_ID="$(ts)-$$"

declare -A SIZE SHA STATUS SCANSUM
CUR_IMAGES=()
PACKS=(); PACK_BYTES=()
DL_SHA=""

on_exit() {
  local rc=$?
  trap - EXIT
  if (( rc != 0 )); then
    err "aborted (exit $rc)"
    (( ${#CUR_IMAGES[@]} )) && docker image rm -f "${CUR_IMAGES[@]}" >/dev/null 2>&1 || true
  fi
  (( KEEP )) || rm -rf "$CTX"
  rm -f "$BUILD_META"
  exit $rc
}
trap on_exit EXIT
trap 'die "interrupted"' INT TERM

# ------------------------------------------------------------------ jq lib --
# The Hub's per-file security object is only loosely specified: it has appeared
# as `security` and as `securityFileStatus`, has been re-wrapped under an `hf`
# key (which broke huggingface_hub in 2024), reports either `status` or `safe`,
# and gains new scanner sub-objects over time. So we keep the raw JSON verbatim
# for the audit, and normalise *generically* by walking whatever scanner
# sub-objects are present rather than hard-coding vendor names.
JQLIB="$WORKDIR/hf2docker.jq"
cat > "$JQLIB" <<'JQEOF'
# Locate the security object and flatten an `hf` wrapper up one level while
# preserving sibling scanners (e.g. third-party vendors alongside `hf`).
def _sec:
  ( .securityFileStatus // .security // null ) as $s
  | if ($s | type) != "object" then null
    elif ($s | has("hf")) and (($s.hf | type) == "object")
      then ($s + $s.hf | del(.hf))
    else $s
    end;

# Best-effort verdict for one scanner sub-object.
def _verdict:
  ( .status?
    // .highestSafetyLevel?
    // (if .virusFound == true then "virus-found"
        elif .virusFound == false then "clean"
        else null end) );

def normalize:
  . as $f
  | ($f | _sec) as $sec
  | ($sec // {}) as $s
  | {
      path:             $f.path,
      size:             ($f.lfs.size // $f.size // 0),
      blob_oid:         ($f.oid // null),
      sha256:           ($f.lfs.oid // $f.lfs.sha256 // null),
      lfs_pointer_size: ($f.lfs.pointerSize // null),
      xet_hash:         ($f.xetHash // $f.xet.hash // null),
      last_commit:      ($f.lastCommit // null),
      security_status:  ( $s.status
                          // (if   $s.safe == true  then "safe"
                              elif $s.safe == false then "unsafe"
                              else null end) ),
      scanners: [ $s | to_entries[]
                  | select((.value | type) == "object")
                  | { name: .key, verdict: (.value | _verdict), details: .value } ],
      security_raw: ($f.securityFileStatus // $f.security // null)
    };

# Merge the tree and paths-info views per path. Later sources must not clobber
# earlier non-null values with nulls or empty arrays, so drop those first.
def _significant: with_entries(select(.value != null and .value != []));

def _defaults:
  { path: null, size: 0, blob_oid: null, sha256: null, lfs_pointer_size: null,
    xet_hash: null, last_commit: null, security_status: null, scanners: [],
    security_raw: null };

def files_from($tree; $paths):
  ( [ $tree[]?  | select((.type // "file") == "file") ]
  + [ $paths[]? | select((.type // "file") == "file") ] )
  | map(normalize)
  | group_by(.path)
  | map( reduce .[] as $x ({}; . * ($x | _significant)) )
  | map(_defaults + .)
  | sort_by(.path);
JQEOF
JQMOD=$(basename "$JQLIB" .jq)
JQDIR=$(dirname "$JQLIB")

# ------------------------------------------------------------- utilities --
slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9._-]+/-/g; s/^[-._]+//; s/[-._]+$//'
}

fsize() {
  if stat -c %s "$1" >/dev/null 2>&1; then stat -c %s "$1"; else stat -f %z "$1"; fi
}

sha256_of() {
  (( LOCAL_SHA )) || { printf ''; return 0; }
  "${SHA_CMD[@]}" "$1" | awk '{print $1}'
}

free_bytes() {
  local d=$1
  while [[ ! -d $d && $d != / ]]; do d=$(dirname "$d"); done
  df -Pk "$d" 2>/dev/null | awk 'NR==2 {printf "%.0f", $4*1024}'
}

reclaim() {
  (( PRUNE )) || return 0
  log "reclaiming disk: pruning build cache + dangling images"
  docker builder prune -af >/dev/null 2>&1 || true
  docker image prune  -f  >/dev/null 2>&1 || true
}

# Ensure room for the group's files AND the image layer Docker will write, plus slack.
require_space() {
  local payload=$1 what=${2:-payload} need attempt=0 avail ok d
  need=$(( payload * SPACE_PCT / 100 + RESERVE_MB * 1024 * 1024 ))
  while :; do
    ok=1
    for d in "$WORKDIR" ${DOCKER_ROOT:+"$DOCKER_ROOT"}; do
      [[ -d $d ]] || continue
      avail=$(free_bytes "$d"); avail=${avail:-0}
      if (( avail < need )); then
        ok=0
        warn "$d: need $(human "$need") for $what, only $(human "$avail") free"
      fi
    done
    (( ok )) && return 0
    (( attempt >= 1 )) && return 1
    attempt=1
    reclaim
  done
}

is_weight() {
  local f=${1##*/} g
  for g in $WEIGHT_GLOBS; do
    # shellcheck disable=SC2053
    [[ ${f,,} == $g ]] && return 0
  done
  return 1
}

hf_curl() {
  local -a auth=()
  [[ -n ${HF_TOKEN:-} ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 \
       ${auth[@]+"${auth[@]}"} "$@"
}

resolve_rev() {
  local repo=$1 sha
  sha=$(hf_curl "${HF_ENDPOINT}/api/models/${repo}/revision/${REVISION}" \
        | jq -r '.sha // empty' 2>/dev/null) || true
  [[ -n ${sha:-} ]] && printf '%s' "$sha" || printf '%s' "$REVISION"
}

# ------------------------------------------------------- audit collection --
# collect_audit <repo> <rev> <outdir>
collect_audit() {
  local repo=$1 rev=$2 out=$3
  mkdir -p "$out"
  log "collecting audit bundle -> $out"

  # --- 1. recursive tree with expansions (carries securityFileStatus, lastCommit)
  local tmp="$out/.tree.raw"
  if ! hf_curl "${HF_ENDPOINT}/api/models/${repo}/tree/${rev}?recursive=1&expand=1" > "$tmp"; then
    warn "expand=1 tree failed; retrying without expansions (scan data will be absent)"
    hf_curl "${HF_ENDPOINT}/api/models/${repo}/tree/${rev}?recursive=1" > "$tmp" \
      || { err "cannot list $repo"; return 1; }
  fi
  jq '.' "$tmp" > "$out/tree.json" || { err "tree response was not valid JSON"; return 1; }
  rm -f "$tmp"

  # --- 2. repo-level info, including the aggregate security status
  if ! hf_curl "${HF_ENDPOINT}/api/models/${repo}?securityStatus=true&revision=${rev}" \
       | jq '.' > "$out/model-info.json" 2>/dev/null; then
    warn "model-info unavailable"; echo '{}' > "$out/model-info.json"
  fi

  # --- 3. paths-info: authoritative per-path record, POSTed in chunks
  local pdir="$out/.paths"; rm -rf "$pdir"; mkdir -p "$pdir"
  jq -r '.[] | select((.type // "file") == "file") | .path' "$out/tree.json" \
    | split -l "$PATHS_CHUNK" - "$pdir/chunk."
  local -a auth=()
  [[ -n ${HF_TOKEN:-} ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  local chunk ok=1
  shopt -s nullglob
  for chunk in "$pdir"/chunk.*; do
    jq -Rs '{paths: (split("\n") | map(select(length > 0))), expand: true}' \
       < "$chunk" > "$chunk.req"
    if ! curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 30 \
              -X POST -H 'Content-Type: application/json' \
              ${auth[@]+"${auth[@]}"} --data "@$chunk.req" \
              "${HF_ENDPOINT}/api/models/${repo}/paths-info/${rev}" > "$chunk.res"; then
      ok=0; break
    fi
  done
  shopt -u nullglob
  if (( ok )) && compgen -G "$pdir/chunk.*.res" >/dev/null; then
    jq -s 'add // []' "$pdir"/chunk.*.res > "$out/paths-info.json"
  else
    warn "paths-info unavailable; relying on tree data alone"
    echo '[]' > "$out/paths-info.json"
  fi
  rm -rf "$pdir"

  # --- 4. normalise both sources into one per-file record
  jq -L "$JQDIR" \
     --slurpfile tree  "$out/tree.json" \
     --slurpfile paths "$out/paths-info.json" \
     -n "include \"${JQMOD}\"; files_from(\$tree[0]; \$paths[0])" \
     > "$out/files.json" || { err "normalisation failed"; return 1; }

  jq -c '.[]' "$out/files.json" > "$out/files.jsonl"

  { printf 'path\tsize\tsha256\tblob_oid\tscan_status\tscanners\n'
    jq -r '.[] | [ .path, .size, (.sha256 // ""), (.blob_oid // ""),
                   (.security_status // "unscanned"),
                   ([ .scanners[]? | "\(.name)=\(.verdict // "n/a")" ] | join(";")) ]
           | @tsv' "$out/files.json"
  } > "$out/files.tsv"

  jq -r '.[] | select(.sha256 != null) | "\(.sha256)  \(.path)"' \
     "$out/files.json" > "$out/SHA256SUMS"

  [[ -f $out/receipts.jsonl ]] || : > "$out/receipts.jsonl"
  [[ -f $out/images.tsv ]] || printf 'run\timage\tdigest\tgroup\tfiles\tbytes\n' > "$out/images.tsv"
  [[ -f $out/groups.tsv ]] || printf 'run\tgroup\tfile\tbytes\timage\n' > "$out/groups.tsv"

  jq -n --arg repo "$repo" --arg rev "$rev" --arg req "$REVISION" \
        --arg run "$RUN_ID" --arg ts "$(ts)" --arg tool "$PROG $VERSION" \
        --arg host "$(hostname 2>/dev/null || echo unknown)" --arg user "${USER:-unknown}" \
        --arg endpoint "$HF_ENDPOINT" \
        --arg base "$BASE_IMAGE" --arg basedigest "$BASE_DIGEST" --arg baseref "$BASE_REF" \
        --argjson files "$(jq 'length' "$out/files.json")" \
        --argjson unscanned "$(jq '[ .[] | select(.security_status == null) ] | length' "$out/files.json")" \
        --argjson unsafe "$(jq '[ .[] | select(.security_status != null and .security_status != "safe") ] | length' "$out/files.json")" \
        '{repo:$repo, requested_revision:$req, commit:$rev, endpoint:$endpoint,
          run:$run, collected_at:$ts, tool:$tool, collected_by:$user, collected_on:$host,
          base_image:$base, base_digest:$basedigest, base_ref:$baseref,
          file_count:$files, files_unscanned:$unscanned, files_not_safe:$unsafe}' \
     > "$out/audit.json"

  log "audit: $(jq -r '"\(.file_count) files, \(.files_not_safe) not-safe, \(.files_unscanned) unscanned"' "$out/audit.json")"
}

# scan_gate <auditdir> <path>  -> 0 allow / 1 block; records STATUS + SCANSUM
scan_gate() {
  local out=$1 path=$2 st sc
  st=$(jq -r --arg p "$path" \
        '.[] | select(.path==$p) | .security_status // "unscanned"' "$out/files.json")
  sc=$(jq -r --arg p "$path" \
        '.[] | select(.path==$p) | [ .scanners[]? | "\(.name)=\(.verdict // "n/a")" ] | join(";")' \
        "$out/files.json")
  STATUS["$path"]=${st:-unscanned}
  SCANSUM["$path"]=${sc:-}

  case ${st:-unscanned} in
    safe) return 0 ;;
    unscanned|null|"")
      warn "no scan result recorded for $path"
      (( REQUIRE_SCANNED )) && return 1
      return 0 ;;
    *)
      warn "scan status for $path is '${st}' [${sc}]"
      (( REQUIRE_SAFE )) && return 1
      return 0 ;;
  esac
}

# ------------------------------------------------------------- networking --
# head_receipt <repo> <rev> <path> -> compact JSON of the interesting headers
head_receipt() {
  local repo=$1 rev=$2 path=$3
  local -a auth=()
  [[ -n ${HF_TOKEN:-} ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  curl -fsSLI --connect-timeout 20 ${auth[@]+"${auth[@]}"} \
       "${HF_ENDPOINT}/${repo}/resolve/${rev}/${path}" 2>/dev/null \
    | awk '
      {
        line=$0; sub(/\r$/,"",line)
        n=index(line,":"); if (n==0) next
        k=tolower(substr(line,1,n-1)); v=substr(line,n+2)
        gsub(/^[ \t]+|[ \t]+$/,"",v); gsub(/["\\]/,"",v)
        if (k=="etag"||k=="x-linked-etag"||k=="x-linked-size"||k=="content-length"||
            k=="x-repo-commit"||k=="last-modified"||k=="x-xet-hash") {
          if (!(k in seen)) { order[++cnt]=k; seen[k]=1 }
          val[k]=v
        }
      }
      END{
        printf "{"
        for (i=1;i<=cnt;i++) printf "%s\"%s\":\"%s\"", (i>1?",":""), order[i], val[order[i]]
        printf "}"
      }' || printf '{}'
}

# download <repo> <rev> <path> <dest> <expected-bytes> <expected-sha>
# Sets the global DL_SHA to the locally computed sha256 (empty if disabled).
download() {
  local repo=$1 rev=$2 path=$3 dest=$4 want=$5 wsha=$6
  local url="${HF_ENDPOINT}/${repo}/resolve/${rev}/${path}" have i
  local -a auth=()
  [[ -n ${HF_TOKEN:-} ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
  DL_SHA=""
  mkdir -p "$(dirname "$dest")"

  have=0; [[ -f $dest ]] && have=$(fsize "$dest")
  if (( want > 0 && have == want )); then
    log "already complete: $path"
  else
    for (( i=1; i<=RETRIES; i++ )); do
      log "GET $path ($(human "$want")) attempt $i/$RETRIES"
      if curl -fL --progress-bar --continue-at - \
              --retry 3 --retry-delay 5 --connect-timeout 20 \
              ${auth[@]+"${auth[@]}"} -o "$dest" "$url"; then
        break
      fi
      (( i == RETRIES )) && return 1
      sleep $(( i * 5 ))
    done
  fi

  have=$(fsize "$dest")
  if (( want > 0 && have != want )); then
    err "size mismatch for $path: got $have, expected $want"
    return 1
  fi

  DL_SHA=$(sha256_of "$dest")
  if [[ -n $DL_SHA && -n $wsha ]] && [[ $DL_SHA != "${wsha#sha256:}" ]]; then
    err "sha256 mismatch for $path: local $DL_SHA vs upstream ${wsha#sha256:}"
    return 1
  fi
  [[ -n $DL_SHA ]] && log "sha256 verified: $path"
  return 0
}

# receipt_record <auditdir> <repo> <rev> <path> <bytes> <declared-sha> <headers-json>
receipt_record() {
  jq -c -n --arg run "$RUN_ID" --arg at "$(ts)" --arg repo "$2" --arg commit "$3" \
           --arg path "$4" --arg declared "$6" --arg localsha "$DL_SHA" \
           --arg status "${STATUS["$4"]:-unscanned}" --arg scanners "${SCANSUM["$4"]:-}" \
           --argjson bytes "$5" --argjson http "$7" \
    '{run:$run, at:$at, repo:$repo, commit:$commit, path:$path, bytes:$bytes,
      sha256_declared:$declared, sha256_local:$localsha,
      scan_status:$status, scanners:$scanners, http:$http}' >> "$1/receipts.jsonl"
}

remote_exists() {
  docker manifest inspect "$1" >/dev/null 2>&1 \
    || docker buildx imagetools inspect "$1" >/dev/null 2>&1
}

image_digest() {
  local img=$1 repo=${1%%:*} d=""
  if [[ -s $BUILD_META ]]; then
    d=$(jq -r '.["containerimage.digest"] // empty' "$BUILD_META" 2>/dev/null || true)
    [[ -n $d ]] && { printf '%s' "$d"; return 0; }
  fi
  d=$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$img" 2>/dev/null \
      | awk -F@ -v r="$repo" '$1==r {print $2; exit}') || true
  printf '%s' "$d"
}

# build_push <context> <dockerfile> <tag> [<tag> ...]
build_push() {
  local ctx=$1 dockerfile=$2; shift 2
  local -a targs=(); local t
  for t in "$@"; do targs+=(-t "$t"); done
  rm -f "$BUILD_META"
  if (( USE_BUILDX )); then
    docker buildx build --platform "$PLATFORM" --provenance=false \
      --metadata-file "$BUILD_META" \
      --output type=registry -f "$dockerfile" "${targs[@]}" "$ctx"
  else
    docker build --platform "$PLATFORM" -f "$dockerfile" "${targs[@]}" "$ctx"
    CUR_IMAGES=("$@")
    for t in "$@"; do docker push "$t"; done
  fi
}

# ------------------------------------------------------------------ packing --
# pack_groups <shard-path>...   -> fills PACKS (newline-joined paths) and PACK_BYTES
pack_groups() {
  PACKS=(); PACK_BYTES=()
  local cur="" cursz=0 cnt=0 p sz flush
  for p in "$@"; do
    sz=${SIZE["$p"]:-0}
    if (( MAX_IMAGE_BYTES > 0 && sz > MAX_IMAGE_BYTES )); then
      warn "$p is $(human "$sz"), larger than --max-image-size; it gets its own image"
    fi
    flush=0
    if (( cnt > 0 )); then
      (( PER_IMAGE > 0 && cnt >= PER_IMAGE )) && flush=1
      (( MAX_IMAGE_BYTES > 0 && cursz + sz > MAX_IMAGE_BYTES )) && flush=1
    fi
    if (( flush )); then
      PACKS+=("$cur"); PACK_BYTES+=("$cursz"); cur=""; cursz=0; cnt=0
    fi
    cur+="${cur:+$'\n'}$p"
    cursz=$(( cursz + sz ))
    cnt=$(( cnt + 1 ))
  done
  (( cnt > 0 )) && { PACKS+=("$cur"); PACK_BYTES+=("$cursz"); }
  return 0
}

# ------------------------------------------------------------ per-repo work --
process_repo() {
  local repo=$1
  local rev model mslug adir repo_status
  rev=$(resolve_rev "$repo")
  model=${MODEL_NAME:-${repo##*/}}          # upstream casing preserved for the path
  mslug=$(slug "$model" | cut -c1-100)      # lowercase, registry-safe, for the tag
  adir="$AUDIT_DIR/$(slug "${repo//\//__}")/${rev}"

  log "==== $repo @ ${rev:0:12}"
  log "     payload -> ${MODELS_ROOT}/${model}"
  log "     images  -> ${IMAGE_REPO}:${TAG_PREFIX}${mslug}-${SHARD_WORD}<N>"
  collect_audit "$repo" "$rev" "$adir" || { err "audit collection failed for $repo"; return 1; }

  repo_status=$(jq -r 'if (.securityStatus | type) == "object"
                       then (.securityStatus.status // "unknown")
                       else (.securityStatus // "unknown" | tostring) end' \
                   "$adir/model-info.json" 2>/dev/null || echo unknown)
  log "repo-level security status: $repo_status"

  SIZE=(); SHA=(); STATUS=(); SCANSUM=()
  local -a shards=() metas=()
  local path sz sha
  while IFS=$'\t' read -r path sz sha; do
    [[ -n $path ]] || continue
    SIZE["$path"]=${sz:-0}
    SHA["$path"]=$sha
    if [[ ${path,,} == *.safetensors ]]; then
      shards+=("$path")
    elif ! is_weight "$path" && (( ${sz:-0} <= META_MAX_MB * 1024 * 1024 )); then
      metas+=("$path")
    fi
  done < <(jq -r '.[] | [ .path, .size, (.sha256 // "") ] | @tsv' "$adir/files.json" \
           | LC_ALL=C sort -t$'\t' -k1,1)

  (( ${#shards[@]} )) || { warn "no .safetensors in $repo — skipped"; return 0; }

  local total=0 meta_bytes=0
  for path in "${shards[@]}"; do total=$(( total + SIZE["$path"] )); done
  for path in ${metas[@]+"${metas[@]}"}; do meta_bytes=$(( meta_bytes + SIZE["$path"] )); done

  pack_groups "${shards[@]}"
  local n=${#PACKS[@]} peak=0 g
  for g in "${PACK_BYTES[@]}"; do (( g > peak )) && peak=$g; done
  (( PACK_BYTES[0] + meta_bytes > peak )) && peak=$(( PACK_BYTES[0] + meta_bytes ))

  log "${#shards[@]} shard(s), $(human "$total"); ${#metas[@]} meta file(s), $(human "$meta_bytes")"
  log "packed into $n image(s) (per-image=${PER_IMAGE}, max=$( (( MAX_IMAGE_BYTES )) && human "$MAX_IMAGE_BYTES" || echo none ))"
  log "largest group $(human "$peak"); peak local disk needed ~$(human $(( peak * SPACE_PCT / 100 + RESERVE_MB*1024*1024 )) )"

  # Apply the scan policy to everything we intend to ship, before any bytes move.
  local blocked=0
  for path in "${shards[@]}" ${metas[@]+"${metas[@]}"}; do
    scan_gate "$adir" "$path" || { err "blocked by scan policy: $path"; blocked=1; }
  done
  if (( blocked )); then
    err "refusing to build from $repo — see $adir/files.tsv"
    return 1
  fi

  local i tag image digest receipt m f gstatus payload dest pdir
  local -a gfiles tags
  for (( i=1; i<=n; i++ )); do
    mapfile -t gfiles <<< "${PACKS[$((i-1))]}"
    tag="${TAG_PREFIX}${mslug}-${SHARD_WORD}${i}"
    tag=${tag:0:128}
    image="${IMAGE_REPO}:${tag}"
    tags=("$image")
    if (( i == 1 )) && [[ -n $FIRST_ALIAS ]]; then
      tags+=("${IMAGE_REPO}:${TAG_PREFIX}${mslug}-$(slug "$FIRST_ALIAS")")
    fi

    payload=${PACK_BYTES[$((i-1))]}
    (( i == 1 )) && payload=$(( payload + meta_bytes ))

    if (( DRY_RUN )); then
      printf '  [%d/%d] %s  (%s%s)\n' "$i" "$n" "${tags[*]}" \
        "$(human "$payload")" "$( (( i == 1 )) && printf ', +%d meta +audit' "${#metas[@]}")"
      for f in "${gfiles[@]}"; do
        printf '          %-46s %10s  scan=%s\n' "$f" \
          "$(human "${SIZE["$f"]}")" "${STATUS["$f"]}"
      done
      continue
    fi

    if (( SKIP_EXISTING )) && remote_exists "$image"; then
      log "[$i/$n] $image already in registry — skipped"
      continue
    fi

    require_space "$payload" "group $i/$n" \
      || die "not enough free disk for group $i ($(human "$payload")); lower --per-image/--max-image-size"

    rm -rf "$CTX"
    pdir="$CTX/payload/$model"          # -> ${MODELS_ROOT}/${model} in the image
    mkdir -p "$pdir"
    : > "$CTX/group-files.jsonl"

    gstatus=safe
    for f in "${gfiles[@]}"; do
      receipt=$(head_receipt "$repo" "$rev" "$f")
      dest="$pdir/$(basename "$f")"
      download "$repo" "$rev" "$f" "$dest" "${SIZE["$f"]}" "${SHA["$f"]}" \
        || die "download failed: $f"
      receipt_record "$adir" "$repo" "$rev" "$f" "${SIZE["$f"]}" "${SHA["$f"]}" "$receipt"
      [[ ${STATUS["$f"]} == safe ]] || gstatus="mixed"
      jq -c -n --arg path "$f" --arg declared "${SHA["$f"]}" --arg localsha "$DL_SHA" \
               --arg status "${STATUS["$f"]}" --arg scanners "${SCANSUM["$f"]}" \
               --argjson bytes "${SIZE["$f"]}" \
        '{path:$path, bytes:$bytes, sha256_declared:$declared, sha256_local:$localsha,
          scan_status:$status, scanners:$scanners}' >> "$CTX/group-files.jsonl"
      printf '%s\t%s\t%s\t%s\t%s\n' "$RUN_ID" "$i" "$f" "${SIZE["$f"]}" "$image" \
        >> "$adir/groups.tsv"
    done

    if (( i == 1 )); then
      for m in ${metas[@]+"${metas[@]}"}; do
        receipt=$(head_receipt "$repo" "$rev" "$m")
        if download "$repo" "$rev" "$m" "$pdir/$m" "${SIZE["$m"]}" "${SHA["$m"]}"; then
          receipt_record "$adir" "$repo" "$rev" "$m" "${SIZE["$m"]}" "${SHA["$m"]}" "$receipt"
        else
          warn "meta file failed, continuing: $m"
        fi
      done
      if (( EMBED_AUDIT )); then
        mkdir -p "$pdir/.audit"
        for f in audit.json files.json files.jsonl files.tsv SHA256SUMS \
                 tree.json paths-info.json model-info.json receipts.jsonl groups.tsv; do
          [[ -f "$adir/$f" ]] && cp "$adir/$f" "$pdir/.audit/"
        done
      fi
    fi

    jq -n \
      --slurpfile files "$CTX/group-files.jsonl" \
      --arg repo "$repo" --arg model "$model" --arg rev "$rev" --arg image "$image" \
      --arg install "${MODELS_ROOT}/${model}" --arg built "$(ts)" \
      --arg tool "$PROG $VERSION" --arg run "$RUN_ID" --arg status "$gstatus" \
      --argjson idx "$i" --argjson cnt "$n" \
      --argjson bytes "${PACK_BYTES[$((i-1))]}" \
      --argjson meta "$( (( i == 1 )) && echo true || echo false )" \
      --argjson metafiles "$(printf '%s\n' ${metas[@]+"${metas[@]}"} | jq -R . | jq -s 'map(select(length>0))')" \
      '{repo:$repo, model:$model, revision:$rev, image:$image, install_path:$install,
        group_index:$idx, group_count:$cnt, group_bytes:$bytes,
        group_scan_status:$status, includes_metadata:$meta,
        files:$files, metadata_files:(if $meta then $metafiles else [] end),
        built:$built, built_by:$tool, run:$run}' \
      > "$pdir/.hf2docker.json"
    rm -f "$CTX/group-files.jsonl"

    { echo "FROM ${BASE_REF}"
      echo "LABEL org.opencontainers.image.title=\"${model} ${SHARD_WORD}${i}/${n}\""
      echo "LABEL org.opencontainers.image.description=\"${#gfiles[@]} safetensors file(s) from ${repo} at ${MODELS_ROOT}/${model}\""
      echo "LABEL org.opencontainers.image.source=\"${HF_ENDPOINT}/${repo}\""
      echo "LABEL org.opencontainers.image.revision=\"${rev}\""
      echo "LABEL org.opencontainers.image.created=\"$(ts)\""
      echo "LABEL io.huggingface.repo=\"${repo}\""
      echo "LABEL io.huggingface.model=\"${model}\""
      echo "LABEL io.huggingface.install_path=\"${MODELS_ROOT}/${model}\""
      echo "LABEL io.huggingface.group=\"${i}/${n}\""
      echo "LABEL io.huggingface.file_count=\"${#gfiles[@]}\""
      echo "LABEL io.huggingface.files=\"$(IFS=,; echo "${gfiles[*]}")\""
      (( ${#gfiles[@]} == 1 )) && \
        echo "LABEL io.huggingface.sha256=\"${SHA["${gfiles[0]}"]}\""
      echo "LABEL io.huggingface.scan_status=\"${gstatus}\""
      echo "LABEL io.huggingface.includes_metadata=\"$( (( i == 1 )) && echo yes || echo no )\""
      echo "LABEL io.hf2docker.base_image=\"${BASE_IMAGE}\""
      [[ -n $BASE_DIGEST ]] && echo "LABEL io.hf2docker.base_digest=\"${BASE_DIGEST}\""
      echo "COPY payload ${MODELS_ROOT}"
      if (( HAS_SHELL )); then
        # Shell into it with: docker run -it --rm <image>
        echo "WORKDIR ${MODELS_ROOT}/${model}"
        echo "CMD [\"${SHELL_PATH}\"]"
      fi
    } > "$CTX/Dockerfile"

    log "[$i/$n] building + pushing ${tags[*]} (${#gfiles[@]} file(s), $(human "$payload"))"
    build_push "$CTX" "$CTX/Dockerfile" "${tags[@]}"

    digest=$(image_digest "$image")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$RUN_ID" "$image" "${digest:-unknown}" \
      "$i/$n" "${#gfiles[@]}" "$payload" >> "$adir/images.tsv"
    for f in "${gfiles[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RUN_ID" "$repo" "$rev" "$model" "$f" "${SIZE["$f"]}" "${SHA["$f"]}" \
        "$image" "${digest:-unknown}" "$i/$n" \
        "$( (( i == 1 )) && echo yes || echo no )" "${STATUS["$f"]}" >> "$INVENTORY"
    done

    # ---- reclaim before moving to the next group ---------------------------
    if (( KEEP )); then
      warn "--keep: leaving $CTX in place (disk bound no longer guaranteed)"
    else
      rm -rf "$CTX"
    fi
    (( ${#CUR_IMAGES[@]} )) && docker image rm -f "${CUR_IMAGES[@]}" >/dev/null 2>&1 || true
    CUR_IMAGES=()
    rm -f "$BUILD_META"
    reclaim
    log "[$i/$n] done; $(human "$(free_bytes "$WORKDIR")") free on $WORKDIR"
  done

  log "audit bundle retained at $adir"
}

# ---------------------------------------------------------------- main loop --
log "$PROG $VERSION — ${#REPOS[@]} repo(s), images -> ${IMAGE_REPO}, workdir $WORKDIR$( (( DRY_RUN )) && echo ' [DRY RUN]')"
failed=()
for repo in "${REPOS[@]}"; do
  if ! process_repo "$repo"; then
    err "repo failed: $repo"
    failed+=("$repo")
  fi
done

if (( ${#failed[@]} )); then
  err "failed repos: ${failed[*]}"
  exit 1
fi
(( DRY_RUN )) || log "all done — inventory in $INVENTORY"
