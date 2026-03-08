#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  echo "Error: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./repo_bundle.sh pack [archive_name.tar.gz]
  ./repo_bundle.sh unpack <archive_name.tar.gz>

What gets packed:
  - tracked files
  - untracked files that are NOT ignored

What gets excluded:
  - .git
  - anything ignored by .gitignore / .git/info/exclude / global git excludes
  - ignored dirs like .venv, logs, outputs, caches, etc.

Examples:
  ./repo_bundle.sh pack
  ./repo_bundle.sh pack exp_snapshot.tar.gz
  ./repo_bundle.sh unpack exp_snapshot.tar.gz
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_NAME="$(basename -- "$REPO_ROOT")"

ensure_git_repo() {
  git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "This script must live under a Git repo (expected repo/scripts)."
}

abs_path() {
  local p="$1"
  if [[ -d "$p" ]]; then
    (cd -- "$p" && pwd -P)
  else
    local d
    d="$(cd -- "$(dirname -- "$p")" && pwd -P)"
    printf '%s/%s\n' "$d" "$(basename -- "$p")"
  fi
}

resolve_archive() {
  local input="$1"

  if [[ -f "$input" ]]; then
    abs_path "$input"
    return 0
  fi

  if [[ -f "${REPO_ROOT}/$input" ]]; then
    abs_path "${REPO_ROOT}/$input"
    return 0
  fi

  die "Archive not found: $input"
}

pack_repo() {
  ensure_git_repo

  local name="${1:-${REPO_NAME}_$(date +%Y%m%d_%H%M%S).tar.gz}"
  name="$(basename -- "$name")"
  [[ "$name" == *.tar.gz ]] || name="${name}.tar.gz"

  local out="${REPO_ROOT}/${name}"
  local tmp_out
  tmp_out="$(mktemp "${TMPDIR:-/tmp}/${REPO_NAME}.XXXXXX.tar.gz")"

  # Build archive from Git's view of the working tree:
  # - tracked files
  # - untracked, non-ignored files
  #
  # Excludes ignored files automatically.
  if ! git -C "$REPO_ROOT" ls-files --cached --others --exclude-standard -z \
      | tar -C "$REPO_ROOT" --null -T - -czf "$tmp_out"; then
    rm -f -- "$tmp_out"
    die "Failed to create archive"
  fi

  mv -f -- "$tmp_out" "$out"
  echo "Created: $out"
}

do_unpack() {
  local archive_input="$1"
  local repo_root="$2"

  local archive
  archive="$(resolve_archive "$archive_input")"

  local tmpdir
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/repo_unpack.XXXXXX")"

  local quoted_tmpdir
  printf -v quoted_tmpdir '%q' "$tmpdir"
  trap "rm -rf -- $quoted_tmpdir" EXIT

  tar -tzf "$archive" >/dev/null || die "Invalid tar.gz archive: $archive"
  tar -xzf "$archive" -C "$tmpdir"

  local src="$tmpdir"
  shopt -s dotglob nullglob
  local top=("$tmpdir"/*)

  # Accept either:
  # - archive with files at root
  # - archive wrapped in a single top-level folder
  if (( ${#top[@]} == 1 )) && [[ -d "${top[0]}" ]]; then
    src="${top[0]}"
  fi

  # Don't stay inside the repo while replacing it
  cd /

  # Remove everything except .git
  local path base
  for path in "$repo_root"/* "$repo_root"/.[!.]* "$repo_root"/..?*; do
    [[ -e "$path" ]] || continue
    base="$(basename -- "$path")"

    if [[ "$base" == ".git" ]]; then
      continue
    fi

    rm -rf -- "$path"
  done

  # Restore archive contents into repo root
  tar -C "$src" -cf - . | tar -C "$repo_root" -xpf -

  echo "Restored repo contents from: $archive"
  echo "Repo root: $repo_root"
}

reexec_unpack() {
  local archive_input="$1"

  local tmp_runner
  tmp_runner="$(mktemp "${TMPDIR:-/tmp}/repo_bundle_runner.XXXXXX.sh")"
  cp -- "$0" "$tmp_runner"
  chmod +x "$tmp_runner"

  exec bash "$tmp_runner" __do_unpack "$archive_input" "$REPO_ROOT"
}

cmd="${1:-}"
case "$cmd" in
  pack)
    pack_repo "${2:-}"
    ;;
  unpack)
    [[ $# -ge 2 ]] || die "unpack requires an archive name or path"
    reexec_unpack "$2"
    ;;
  __do_unpack)
    [[ $# -eq 3 ]] || die "internal unpack invocation failed"
    do_unpack "$2" "$3"
    rm -f -- "$0"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    die "Unknown command: $cmd"
    ;;
esac