#!/usr/bin/env bash
# =============================================================================
#  hf-audit.sh — provenance and scan audit reports for Hugging Face model repos
# =============================================================================
#
#  The audit half of hf2docker.sh, on its own. It downloads no weights, builds
#  no images, and needs no Docker daemon — only curl and jq — so it can run in a
#  CI job, on a laptop, or against repos you have no intention of packaging.
#
#  For each repo it:
#    1. Resolves the revision to an immutable commit SHA.
#    2. Fetches and retains the raw upstream JSON verbatim (tree, paths-info,
#       model-info) as the primary evidence.
#    3. Normalises it into a per-file record: size, git blob oid, LFS sha256,
#       Xet hash, last commit, and every scanner verdict the Hub reports
#       (antivirus, pickle import scan, third-party scanners, ...).
#    4. Writes SHA256SUMS, a flat TSV, and a human-readable report.md.
#    5. Optionally applies a scan policy and exits non-zero, so it can gate a
#       pipeline before anything downloads gigabytes.
#
#  The normaliser is byte-identical to the one in hf2docker.sh (see
#  NORMALIZER_VERSION), so records produced by either tool are comparable.
#
#  Re-running against an existing bundle regenerates the derived files from the
#  retained raw JSON without touching the network. Use --refresh to re-fetch;
#  the previous raw JSON is archived under history/<timestamp>/ rather than
#  overwritten, because a scan verdict changing over time is itself evidence.
#
#  Requirements: bash >= 4, curl, jq, awk, sed, sort, split
#  Optional env: HF_TOKEN (gated/private repos), HF_ENDPOINT
#
#  Examples:
#    ./hf-audit.sh Qwen/Qwen3-Coder-Next-FP8
#    ./hf-audit.sh -f repos.txt -o /srv/audit --require-safe
#    ./hf-audit.sh --plan --per-image 3 --max-image-size 6GB MERaLiON/MERaLiON-SpeechEncoder-2
#
#  Exit codes: 0 ok   1 error   2 scan policy violation
#
# =============================================================================
set -Eeuo pipefail

PROG=${0##*/}
VERSION=1.0.0
NORMALIZER_VERSION=1        # bump only when the jq block below changes

# ------------------------------------------------------------------ defaults --
OUTDIR=${OUTDIR:-${PWD}/hf-audit}
REVISION=${REVISION:-main}
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
PATHS_CHUNK=${PATHS_CHUNK:-500}
REPORT=${REPORT:-md}          # md | none
MAX_TABLE_ROWS=${MAX_TABLE_ROWS:-60}
REFRESH=0
REQUIRE_SAFE=0
REQUIRE_SCANNED=0

# Optional "what would hf2docker build" preview
PLAN=0
PER_IMAGE=${PER_IMAGE:-1}
MAX_IMAGE_SIZE=${MAX_IMAGE_SIZE:-}
MAX_IMAGE_BYTES=0
DOCKER_NS=${DOCKER_NS:-}
IMAGE_NAME=${IMAGE_NAME:-mlbakery}
MODELS_ROOT=${MODELS_ROOT:-/models}
SHARD_WORD=${SHARD_WORD:-shard}
TAG_PREFIX=${TAG_PREFIX:-}
META_MAX_MB=${META_MAX_MB:-64}

REPOS=()
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
$PROG $VERSION — provenance and scan audit reports for Hugging Face model repos.
Downloads no weights, builds no images, needs no Docker.

Usage: $PROG [options] <repo> [<repo> ...]

Options:
  -o, --out DIR          Where bundles are written        (default: $OUTDIR)
  -f, --repos-file FILE  Read repo ids from FILE (one per line, # comments ok)
  -r, --revision REV     Branch/tag/commit to audit       (default: $REVISION)
      --refresh          Re-fetch from the Hub even if a bundle exists;
                         the previous raw JSON is archived under history/
      --report md|none   Human-readable report            (default: $REPORT)
      --max-rows N       Row cap for report tables        (default: $MAX_TABLE_ROWS)
      --paths-chunk N    Paths per paths-info POST        (default: $PATHS_CHUNK)

Policy gate (exit code 2 on violation):
      --require-safe     Fail if any file's scan status is not "safe"
      --require-scanned  Also fail if any file has no scan result yet

Packaging preview (what hf2docker.sh would build):
      --plan             Add a planned-images section to the report
      --per-image N      safetensors per image            (default: $PER_IMAGE)
      --max-image-size SZ  Cap group payload, e.g. 6GB    (default: none)
  -n, --namespace NS     Registry namespace for the preview
  -i, --image-name NAME  Image repo name                  (default: $IMAGE_NAME)
      --models-root PATH In-image parent directory        (default: $MODELS_ROOT)
      --shard-word W     Word used in the tag             (default: $SHARD_WORD)
  -t, --tag-prefix S     Prefix added to every tag
      --meta-max-mb N    Max size of a metadata file, MiB (default: $META_MAX_MB)

  -h, --help             This help
  -V, --version          Version

Bundle layout:
  <out>/audit-index.tsv            append-only, one row per repo per run
  <out>/<org>__<name>/<commit>/
      tree.json         raw GET  /api/models/<repo>/tree/<rev>?recursive=1&expand=1
      paths-info.json   raw POST /api/models/<repo>/paths-info/<rev>  (chunked)
      model-info.json   raw GET  /api/models/<repo>?securityStatus=true
      files.json        normalised per-file record
      files.jsonl       same, one object per line, for log/SIEM ingestion
      files.tsv         flat view for grepping / spreadsheets
      SHA256SUMS        coreutils format, usable with 'sha256sum -c'
      report.md         human-readable report
      audit.json        run manifest (tool + normaliser version, who, when)
      history/<ts>/     previous raw JSON, kept when --refresh re-fetches
EOF
}

# ------------------------------------------------------------- arg parsing --
read_repos_file() {
  local f=$1 line n=0
  [[ -r $f ]] || die "cannot read $f"
  # `read` returns non-zero on a final line with no trailing newline, so the
  # plain idiom drops it; `|| [[ -n $line ]]` catches that, and clearing `line`
  # at the end of the body stops the clause re-triggering forever.
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%%#*}
    line=${line//[[:space:]]/}
    if [[ -n $line ]]; then REPOS+=("$line"); n=$(( n + 1 )); fi
    line=""
  done < "$f"
  (( n )) || die "no repositories found in $f"
  log "read $n repo(s) from $f"
}

while (( $# )); do
  case $1 in
    -o|--out|-f|--repos-file|-r|--revision|--report|--max-rows|--paths-chunk|\
    --per-image|--max-image-size|-n|--namespace|-i|--image-name|--models-root|\
    --shard-word|-t|--tag-prefix|--meta-max-mb)
      (( $# >= 2 )) || die "option $1 requires a value"
      if [[ $2 == -?* ]]; then
        die "option $1 requires a value but got '$2' — repeated flag or missing argument?"
      fi
      ;;
  esac
  case $1 in
    -o|--out)           OUTDIR=$2; shift 2 ;;
    -f|--repos-file)    read_repos_file "$2"; shift 2 ;;
    -r|--revision)      REVISION=$2; shift 2 ;;
    --refresh)          REFRESH=1; shift ;;
    --report)           REPORT=$2; shift 2 ;;
    --max-rows)         MAX_TABLE_ROWS=$2; shift 2 ;;
    --paths-chunk)      PATHS_CHUNK=$2; shift 2 ;;
    --require-safe)     REQUIRE_SAFE=1; shift ;;
    --require-scanned)  REQUIRE_SAFE=1; REQUIRE_SCANNED=1; shift ;;
    --plan)             PLAN=1; shift ;;
    --per-image)        PER_IMAGE=$2; PLAN=1; shift 2 ;;
    --max-image-size)   MAX_IMAGE_SIZE=$2; PLAN=1; shift 2 ;;
    -n|--namespace)     DOCKER_NS=$2; shift 2 ;;
    -i|--image-name)    IMAGE_NAME=$2; shift 2 ;;
    --models-root)      MODELS_ROOT=${2%/}; shift 2 ;;
    --shard-word)       SHARD_WORD=$2; shift 2 ;;
    -t|--tag-prefix)    TAG_PREFIX=$2; shift 2 ;;
    --meta-max-mb)      META_MAX_MB=$2; shift 2 ;;
    -h|--help)          usage; exit 0 ;;
    -V|--version)       echo "$PROG $VERSION (normaliser v$NORMALIZER_VERSION)"; exit 0 ;;
    --)                 shift; while (( $# )); do REPOS+=("$1"); shift; done ;;
    -*)                 die "unknown option: $1 (try --help)" ;;
    *)                  REPOS+=("$1"); shift ;;
  esac
done

# -------------------------------------------------------------- preflight --
(( BASH_VERSINFO[0] >= 4 )) || die "bash >= 4 required (found $BASH_VERSION)"
for c in curl jq awk sed sort split; do
  command -v "$c" >/dev/null || die "missing required command: $c"
done
(( ${#REPOS[@]} )) || { usage; die "no repositories given"; }
check_int() { [[ $2 =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer, got '$2'"; }
check_int --per-image   "$PER_IMAGE"
check_int --paths-chunk "$PATHS_CHUNK"
check_int --max-rows    "$MAX_TABLE_ROWS"
check_int --meta-max-mb "$META_MAX_MB"
(( PATHS_CHUNK >= 1 )) || die "--paths-chunk must be at least 1"
case $REPORT in md|none) ;; *) die "--report must be 'md' or 'none'" ;; esac
[[ -n $MAX_IMAGE_SIZE ]] && MAX_IMAGE_BYTES=$(to_bytes "$MAX_IMAGE_SIZE")

mkdir -p "$OUTDIR"
INDEX="$OUTDIR/audit-index.tsv"
IDX_HEADER=$'run\tcollected_at\trepo\tcommit\tfiles\tbytes\tsafetensors\tnot_safe\tunscanned\tbundle'
if [[ -f $INDEX ]]; then
  if [[ $(head -n1 "$INDEX") != "$IDX_HEADER" ]]; then
    mv "$INDEX" "${INDEX%.tsv}-superseded-$(date -u +%Y%m%dT%H%M%SZ).tsv"
    printf '%s\n' "$IDX_HEADER" > "$INDEX"
  fi
else
  printf '%s\n' "$IDX_HEADER" > "$INDEX"
fi
RUN_ID="$(ts)-$$"
VIOLATIONS=0

# ------------------------------------------------------------------ jq lib --
# NOTE: this block is byte-identical to the one in hf2docker.sh. If you change
# one, change both and bump NORMALIZER_VERSION in each, or the two tools will
# silently produce non-comparable records.
#
# The Hub's per-file security object is only loosely specified: it has appeared
# as `security` and as `securityFileStatus`, has been re-wrapped under an `hf`
# key (which broke huggingface_hub in 2024), reports either `status` or `safe`,
# and gains new scanner sub-objects over time. So we keep the raw JSON verbatim
# for the audit, and normalise *generically* by walking whatever scanner
# sub-objects are present rather than hard-coding vendor names.
TMPROOT=$(mktemp -d "${TMPDIR:-/tmp}/hf-audit.XXXXXX")
trap 'rm -rf "$TMPROOT"' EXIT
JQLIB="$TMPROOT/hf2docker.jq"
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

# Report-only helpers, kept separate so the shared block above stays identical.
REPLIB="$TMPROOT/report.jq"
cat > "$REPLIB" <<'JQEOF'
def hsize:
  ( . // 0 ) as $b
  | if   $b >= 1099511627776 then "\(($b/1099511627776*100|round)/100) TiB"
    elif $b >= 1073741824    then "\(($b/1073741824*100|round)/100) GiB"
    elif $b >= 1048576       then "\(($b/1048576*100|round)/100) MiB"
    elif $b >= 1024          then "\(($b/1024*100|round)/100) KiB"
    else "\($b) B" end;

def scanstr: [ .scanners[]? | "\(.name)=\(.verdict // "n/a")" ] | join(", ");
def status_of: .security_status // "unscanned";
def shortsha: if . == null or . == "" then "-" else .[0:16] end;
JQEOF

# ------------------------------------------------------------- utilities --
slug() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9._-]+/-/g; s/^[-._]+//; s/[-._]+$//'
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

# --------------------------------------------------------------- fetching --
# fetch_raw <repo> <rev> <outdir>  — writes tree.json, model-info.json, paths-info.json
fetch_raw() {
  local repo=$1 rev=$2 out=$3
  mkdir -p "$out"

  local tmp="$out/.tree.raw"
  if ! hf_curl "${HF_ENDPOINT}/api/models/${repo}/tree/${rev}?recursive=1&expand=1" > "$tmp"; then
    warn "expand=1 tree failed; retrying without expansions (scan data will be absent)"
    hf_curl "${HF_ENDPOINT}/api/models/${repo}/tree/${rev}?recursive=1" > "$tmp" \
      || { err "cannot list $repo"; return 1; }
  fi
  jq '.' "$tmp" > "$out/tree.json" || { err "tree response was not valid JSON"; return 1; }
  rm -f "$tmp"

  if ! hf_curl "${HF_ENDPOINT}/api/models/${repo}?securityStatus=true&revision=${rev}" \
       | jq '.' > "$out/model-info.json" 2>/dev/null; then
    warn "model-info unavailable"; echo '{}' > "$out/model-info.json"
  fi

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
}

# archive_raw <outdir> — keep the previous evidence before re-fetching
archive_raw() {
  local out=$1 stamp hist f
  stamp=$(jq -r '.collected_at // "unknown"' "$out/audit.json" 2>/dev/null || echo unknown)
  hist="$out/history/${stamp//:/}"
  mkdir -p "$hist"
  for f in tree.json paths-info.json model-info.json files.json audit.json; do
    [[ -f "$out/$f" ]] && cp "$out/$f" "$hist/"
  done
  log "previous bundle archived to $hist"
}

# --------------------------------------------------------------- deriving --
# derive <repo> <rev> <outdir>
derive() {
  local repo=$1 rev=$2 out=$3

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

  jq -n --arg repo "$repo" --arg rev "$rev" --arg req "$REVISION" \
        --arg run "$RUN_ID" --arg ts "$(ts)" --arg tool "$PROG $VERSION" \
        --arg host "$(hostname 2>/dev/null || echo unknown)" --arg user "${USER:-unknown}" \
        --arg endpoint "$HF_ENDPOINT" \
        --argjson normv "$NORMALIZER_VERSION" \
        --argjson files "$(jq 'length' "$out/files.json")" \
        --argjson bytes "$(jq '[ .[].size ] | add // 0' "$out/files.json")" \
        --argjson st "$(jq '[ .[] | select(.path | ascii_downcase | endswith(".safetensors")) ] | length' "$out/files.json")" \
        --argjson unscanned "$(jq '[ .[] | select(.security_status == null) ] | length' "$out/files.json")" \
        --argjson unsafe "$(jq '[ .[] | select(.security_status != null and .security_status != "safe") ] | length' "$out/files.json")" \
        '{repo:$repo, requested_revision:$req, commit:$rev, endpoint:$endpoint,
          run:$run, collected_at:$ts, tool:$tool, normalizer_version:$normv,
          collected_by:$user, collected_on:$host,
          file_count:$files, total_bytes:$bytes, safetensors_count:$st,
          files_unscanned:$unscanned, files_not_safe:$unsafe}' \
     > "$out/audit.json"
}

# ---------------------------------------------------------------- planning --
PACKS=(); PACK_BYTES=()
declare -A SIZE
pack_groups() {
  PACKS=(); PACK_BYTES=()
  local cur="" cursz=0 cnt=0 p sz flush
  for p in "$@"; do
    sz=${SIZE["$p"]:-0}
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

# plan_section <repo> <outdir> -> markdown on stdout
plan_section() {
  local repo=$1 out=$2
  local model mslug imgrepo path sz i n meta_bytes=0
  local -a shards=() metas=() gfiles
  model=${repo##*/}
  mslug=$(slug "$model" | cut -c1-100)
  imgrepo=$IMAGE_NAME
  [[ $IMAGE_NAME != */* && -n $DOCKER_NS ]] && imgrepo="$(slug "$DOCKER_NS")/$IMAGE_NAME"

  SIZE=()
  while IFS=$'\t' read -r path sz; do
    [[ -n $path ]] || continue
    SIZE["$path"]=${sz:-0}
    if [[ ${path,,} == *.safetensors ]]; then
      shards+=("$path")
    elif ! is_weight "$path" && (( ${sz:-0} <= META_MAX_MB * 1024 * 1024 )); then
      metas+=("$path"); meta_bytes=$(( meta_bytes + ${sz:-0} ))
    fi
  done < <(jq -r '.[] | [ .path, .size ] | @tsv' "$out/files.json" | LC_ALL=C sort -t$'\t' -k1,1)

  echo
  echo "## Planned images"
  echo
  if (( ${#shards[@]} == 0 )); then
    echo "No \`.safetensors\` files — nothing would be built."
    return 0
  fi
  pack_groups "${shards[@]}"
  n=${#PACKS[@]}
  printf 'Packing: `--per-image %s`' "$PER_IMAGE"
  [[ -n $MAX_IMAGE_SIZE ]] && printf ', `--max-image-size %s`' "$MAX_IMAGE_SIZE"
  printf '. Payload path `%s/%s`.\n\n' "$MODELS_ROOT" "$model"
  echo "| # | Tag | Files | Payload | Contents |"
  echo "|---|-----|-------|---------|----------|"
  local peak=0
  for (( i=1; i<=n; i++ )); do
    mapfile -t gfiles <<< "${PACKS[$((i-1))]}"
    local pbytes=${PACK_BYTES[$((i-1))]}
    (( i == 1 )) && pbytes=$(( pbytes + meta_bytes ))
    (( pbytes > peak )) && peak=$pbytes
    printf '| %d | `%s:%s%s-%s%d` | %d | %s | %s%s |\n' \
      "$i" "$imgrepo" "$TAG_PREFIX" "$mslug" "$SHARD_WORD" "$i" \
      "${#gfiles[@]}" "$(human "$pbytes")" \
      "$(printf '%s, ' "${gfiles[@]}" | sed 's/, $//')" \
      "$( (( i == 1 )) && printf ' + %d metadata files + audit bundle' "${#metas[@]}")"
  done
  echo
  printf 'Largest group %s, so hf2docker would need roughly **%s** free disk at the default `--space-pct 250`.\n' \
    "$(human "$peak")" "$(human $(( peak * 250 / 100 + 2048*1024*1024 )) )"
}

# ----------------------------------------------------------------- report --
# render_report <repo> <rev> <outdir>
render_report() {
  local repo=$1 rev=$2 out=$3
  local repo_status
  repo_status=$(jq -r 'if (.securityStatus | type) == "object"
                       then (.securityStatus.status // "unknown")
                       else (.securityStatus // "unknown" | tostring) end' \
                   "$out/model-info.json" 2>/dev/null || echo unknown)

  {
    echo "# Audit report — \`$repo\`"
    echo
    jq -r '"| Field | Value |\n|---|---|\n" +
           "| Repository | `\(.repo)` |\n" +
           "| Requested revision | `\(.requested_revision)` |\n" +
           "| Resolved commit | `\(.commit)` |\n" +
           "| Endpoint | \(.endpoint) |\n" +
           "| Files | \(.file_count) |\n" +
           "| Safetensors | \(.safetensors_count) |\n" +
           "| Collected at | \(.collected_at) |\n" +
           "| Collected by | \(.collected_by)@\(.collected_on) |\n" +
           "| Tool | \(.tool), normaliser v\(.normalizer_version) |"' "$out/audit.json"
    echo "| Total size | $(human "$(jq -r '.total_bytes' "$out/audit.json")") |"
    echo "| Repo-level security status | \`$repo_status\` |"
    echo

    echo "## Scan status"
    echo
    echo "| Status | Files | Size |"
    echo "|--------|-------|------|"
    jq -r -L "$JQDIR" 'include "report";
      group_by(status_of)
      | map({s: (.[0] | status_of), n: length, b: ([.[].size] | add // 0)})
      | sort_by(-.n)[]
      | "| `\(.s)` | \(.n) | \(.b | hsize) |"' "$out/files.json"
    echo

    if [[ $(jq '[ .[] | .scanners[]? ] | length' "$out/files.json") != 0 ]]; then
      echo "## Scanner verdicts"
      echo
      echo "| Scanner | Verdict | Files |"
      echo "|---------|---------|-------|"
      jq -r '[ .[] | .scanners[]? | {n: .name, v: (.verdict // "n/a")} ]
             | group_by([.n, .v])
             | map({scanner: .[0].n, verdict: .[0].v, count: length})
             | sort_by(.scanner, .verdict)[]
             | "| `\(.scanner)` | \(.verdict) | \(.count) |"' "$out/files.json"
      echo
    else
      echo "> No per-file scanner results were returned for this revision."
      echo "> Scans may still be pending, or the endpoint may not expose them."
      echo
    fi

    local flagged
    flagged=$(jq '[ .[] | select(.security_status != null and .security_status != "safe") ] | length' \
                 "$out/files.json")
    if (( flagged > 0 )); then
      echo "## Flagged files ($flagged)"
      echo
      echo "| File | Status | Scanners | Size |"
      echo "|------|--------|----------|------|"
      jq -r -L "$JQDIR" --argjson cap "$MAX_TABLE_ROWS" 'include "report";
        [ .[] | select(.security_status != null and .security_status != "safe") ]
        | .[0:$cap][]
        | "| `\(.path)` | **\(.security_status)** | \(scanstr) | \(.size | hsize) |"' \
        "$out/files.json"
      (( flagged > MAX_TABLE_ROWS )) && echo && echo "_… and $(( flagged - MAX_TABLE_ROWS )) more; see \`files.tsv\`._"
      echo
    fi

    local unscanned
    unscanned=$(jq '[ .[] | select(.security_status == null) ] | length' "$out/files.json")
    if (( unscanned > 0 )); then
      echo "## Files with no scan result ($unscanned)"
      echo
      jq -r --argjson cap "$MAX_TABLE_ROWS" \
        '[ .[] | select(.security_status == null) | .path ] | .[0:$cap][] | "- `\(.)`"' \
        "$out/files.json"
      (( unscanned > MAX_TABLE_ROWS )) && echo && echo "_… and $(( unscanned - MAX_TABLE_ROWS )) more._"
      echo
    fi

    echo "## Safetensors"
    echo
    echo "| File | Size | sha256 (first 16) | Status |"
    echo "|------|------|-------------------|--------|"
    jq -r -L "$JQDIR" --argjson cap "$MAX_TABLE_ROWS" 'include "report";
      [ .[] | select(.path | ascii_downcase | endswith(".safetensors")) ]
      | .[0:$cap][]
      | "| `\(.path)` | \(.size | hsize) | `\(.sha256 | shortsha)` | \(status_of) |"' \
      "$out/files.json"
    echo

    if (( PLAN )); then plan_section "$repo" "$out"; fi

    echo
    echo "## Evidence in this bundle"
    echo
    echo "| File | What it is |"
    echo "|------|------------|"
    echo "| \`tree.json\` | raw Hub tree response, verbatim |"
    echo "| \`paths-info.json\` | raw Hub paths-info response, verbatim |"
    echo "| \`model-info.json\` | raw repo metadata incl. aggregate security status |"
    echo "| \`files.json\` / \`files.jsonl\` | normalised per-file record |"
    echo "| \`files.tsv\` | flat view for grep and spreadsheets |"
    echo "| \`SHA256SUMS\` | verify downloads with \`sha256sum -c\` |"
    echo "| \`audit.json\` | run manifest: tool, normaliser version, who, when |"
    echo
    echo "_Generated by $PROG $VERSION at $(ts)._"
  } > "$out/report.md"
}

# ------------------------------------------------------------ per-repo work --
process_repo() {
  local repo=$1 rev out fresh=1
  rev=$(resolve_rev "$repo")
  out="$OUTDIR/$(slug "${repo//\//__}")/${rev}"

  log "==== $repo @ ${rev:0:12}"

  if [[ -f "$out/tree.json" && -f "$out/paths-info.json" ]]; then
    if (( REFRESH )); then
      archive_raw "$out"
    else
      fresh=0
      log "existing bundle found; regenerating derived files offline (--refresh to re-fetch)"
    fi
  fi
  if (( fresh )); then
    log "fetching from $HF_ENDPOINT"
    fetch_raw "$repo" "$rev" "$out" || { err "fetch failed for $repo"; return 1; }
  fi

  derive "$repo" "$rev" "$out" || return 1
  [[ $REPORT == md ]] && render_report "$repo" "$rev" "$out"

  local files bytes st unsafe unscanned
  files=$(jq -r '.file_count' "$out/audit.json")
  bytes=$(jq -r '.total_bytes' "$out/audit.json")
  st=$(jq -r '.safetensors_count' "$out/audit.json")
  unsafe=$(jq -r '.files_not_safe' "$out/audit.json")
  unscanned=$(jq -r '.files_unscanned' "$out/audit.json")

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$RUN_ID" "$(ts)" "$repo" "$rev" "$files" "$bytes" "$st" "$unsafe" "$unscanned" "$out" \
    >> "$INDEX"

  log "$files files, $(human "$bytes"), $st safetensors; $unsafe not-safe, $unscanned unscanned"
  log "bundle: $out"
  [[ $REPORT == md ]] && log "report: $out/report.md"

  if (( REQUIRE_SAFE && unsafe > 0 )); then
    err "$repo: $unsafe file(s) with a non-safe scan status"
    VIOLATIONS=$(( VIOLATIONS + 1 ))
  fi
  if (( REQUIRE_SCANNED && unscanned > 0 )); then
    err "$repo: $unscanned file(s) with no scan result"
    VIOLATIONS=$(( VIOLATIONS + 1 ))
  fi
  return 0
}

# ---------------------------------------------------------------- main loop --
log "$PROG $VERSION — ${#REPOS[@]} repo(s) -> $OUTDIR"
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
if (( VIOLATIONS )); then
  err "scan policy violated in $VIOLATIONS case(s)"
  exit 2
fi
log "all done — index in $INDEX"
