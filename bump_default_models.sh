#!/usr/bin/env bash
# Bump the default strong and eval models used by wdoc.
#
# Source of truth: wdoc/utils/env.py
#   - WDOC_DEFAULT_MODEL              (strong model)
#   - WDOC_DEFAULT_QUERY_EVAL_MODEL   (eval model)
#
# The script reads the current values from env.py, then replaces them
# across the docs / README / architecture files. It replaces both the
# full model id ("provider/path/name") and the basename (everything
# after the last slash), since some files use the short form.
#
# Behavior is dry-run by default. Pass --apply to actually write.
# The script never commits; review with `git diff` and commit yourself.
#
# Usage:
#   ./bump_default_models.sh <NEW_STRONG> <NEW_EVAL>           # preview
#   ./bump_default_models.sh <NEW_STRONG> <NEW_EVAL> --apply   # write

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Walk up from the script's directory until we find the env.py source of
# truth. Lets the script live at repo root or one level down without
# hard-coding the layout.
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -f "$REPO_ROOT/wdoc/utils/env.py" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [ ! -f "$REPO_ROOT/wdoc/utils/env.py" ]; then
  echo "error: could not locate wdoc/utils/env.py from $SCRIPT_DIR upward" >&2
  exit 1
fi
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: bump_default_models.sh <NEW_STRONG_MODEL> <NEW_EVAL_MODEL> [--apply]

Reads current defaults from wdoc/utils/env.py and replaces them in:
  wdoc/utils/env.py, wdoc/docs/help.md, SKILL.md, README.md, ARCHITECTURE.md

Both the full id ("provider/.../name") and the basename are replaced.
A post-flight scan warns about any leftover mention of the old names or
the old model family (the prefix of the basename, e.g. "gemini").

Examples:
  ./bump_default_models.sh deepseek/deepseek-v4-pro deepseek/deepseek-v4-flash
  ./bump_default_models.sh openrouter/anthropic/claude-4 openrouter/anthropic/claude-haiku --apply

Without --apply, only prints the plan.
EOF
}

APPLY=false
POS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --apply)   APPLY=true ;;
    --*)       echo "error: unknown flag: $arg" >&2; usage >&2; exit 2 ;;
    *)         POS+=("$arg") ;;
  esac
done

if [ "${#POS[@]}" -ne 2 ]; then
  usage >&2
  exit 2
fi

NEW_STRONG="${POS[0]}"
NEW_EVAL="${POS[1]}"

ENV_FILE="wdoc/utils/env.py"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found (wrong layout?)" >&2; exit 1; }

# Pull current defaults out of env.py. Uses python so we don't need to
# trust sed/awk with regex escaping.
extract_default() {
  python - "$1" "$ENV_FILE" <<'PY'
import re, sys
key, path = sys.argv[1], sys.argv[2]
pat = re.compile(rf'^\s+{re.escape(key)}\s*:\s*str\s*=\s*"([^"]+)"\s*$')
with open(path) as f:
    for line in f:
        m = pat.match(line)
        if m:
            print(m.group(1))
            sys.exit(0)
sys.exit(1)
PY
}

OLD_STRONG="$(extract_default WDOC_DEFAULT_MODEL)" \
  || { echo "error: could not parse WDOC_DEFAULT_MODEL from $ENV_FILE" >&2; exit 1; }
OLD_EVAL="$(extract_default WDOC_DEFAULT_QUERY_EVAL_MODEL)" \
  || { echo "error: could not parse WDOC_DEFAULT_QUERY_EVAL_MODEL from $ENV_FILE" >&2; exit 1; }

# Basename = everything after the last slash, so doc files using the
# short form (e.g. ARCHITECTURE.md table) get caught.
OLD_STRONG_BN="${OLD_STRONG##*/}"
OLD_EVAL_BN="${OLD_EVAL##*/}"
NEW_STRONG_BN="${NEW_STRONG##*/}"
NEW_EVAL_BN="${NEW_EVAL##*/}"

# Family = basename truncated at the first dash, dot, or digit.
# e.g. gemini-3.1-pro-preview -> gemini; deepseek-v4-pro -> deepseek.
# Used only for a soft post-flight warning.
OLD_STRONG_FAM="${OLD_STRONG_BN%%[-0-9.]*}"
OLD_EVAL_FAM="${OLD_EVAL_BN%%[-0-9.]*}"

if [ "$OLD_STRONG" = "$NEW_STRONG" ] && [ "$OLD_EVAL" = "$NEW_EVAL" ]; then
  echo "nothing to do: requested defaults already match current values"
  exit 0
fi

echo "Plan:"
echo "  strong: $OLD_STRONG  ->  $NEW_STRONG"
echo "          basename: $OLD_STRONG_BN -> $NEW_STRONG_BN"
echo "  eval:   $OLD_EVAL  ->  $NEW_EVAL"
echo "          basename: $OLD_EVAL_BN -> $NEW_EVAL_BN"
echo

FILES=(
  "wdoc/utils/env.py"
  "wdoc/docs/help.md"
  "SKILL.md"
  "README.md"
  "ARCHITECTURE.md"
)

# Env-style files (KEY=VALUE). Replaced via line-anchored regex rather than
# literal string match, so they get re-synced even if they had drifted from
# env.py before the bump.
ENV_FILES=(
  "docker/env.example"
)

for f in "${FILES[@]}" "${ENV_FILES[@]}"; do
  [ -f "$f" ] || { echo "error: missing expected file: $f" >&2; exit 1; }
done

echo "Occurrences in target files (full / basename):"
printf "  %-22s  %-11s  %-11s\n" "file" "strong" "eval"
total_strong=0
total_eval=0
for f in "${FILES[@]}"; do
  s_full=$(grep -Fc -- "$OLD_STRONG"    "$f" || true)
  s_bn=$(  grep -Fc -- "$OLD_STRONG_BN" "$f" || true)
  e_full=$(grep -Fc -- "$OLD_EVAL"      "$f" || true)
  e_bn=$(  grep -Fc -- "$OLD_EVAL_BN"   "$f" || true)
  printf "  %-22s  %d / %-7d  %d / %-7d\n" "$f" "$s_full" "$s_bn" "$e_full" "$e_bn"
  total_strong=$((total_strong + s_full))
  total_eval=$((total_eval + e_full))
done
echo

echo "Env files (current KEY=VALUE in file -> new value):"
for f in "${ENV_FILES[@]}"; do
  cur_s=$(grep -E '^WDOC_DEFAULT_MODEL=' "$f"            | head -n1 | cut -d= -f2- || true)
  cur_e=$(grep -E '^WDOC_DEFAULT_QUERY_EVAL_MODEL=' "$f" | head -n1 | cut -d= -f2- || true)
  echo "  $f"
  echo "    WDOC_DEFAULT_MODEL=${cur_s:-<missing>}  ->  $NEW_STRONG"
  echo "    WDOC_DEFAULT_QUERY_EVAL_MODEL=${cur_e:-<missing>}  ->  $NEW_EVAL"
  if [ -n "$cur_s" ] && [ "$cur_s" != "$OLD_STRONG" ]; then
    echo "    NOTE: env file's current strong value drifted from env.py ($OLD_STRONG) - will be re-synced."
  fi
  if [ -n "$cur_e" ] && [ "$cur_e" != "$OLD_EVAL" ]; then
    echo "    NOTE: env file's current eval value drifted from env.py ($OLD_EVAL) - will be re-synced."
  fi
done
echo

# Safety: refuse to run if we can't even find the old strong/eval names
# anywhere in the target files. Likely means env.py drifted or paths changed.
if [ "$OLD_STRONG" != "$NEW_STRONG" ] && [ "$total_strong" -eq 0 ]; then
  echo "error: zero matches for old strong model in target files; aborting" >&2
  exit 1
fi
if [ "$OLD_EVAL" != "$NEW_EVAL" ] && [ "$total_eval" -eq 0 ]; then
  echo "error: zero matches for old eval model in target files; aborting" >&2
  exit 1
fi

if ! $APPLY; then
  echo "Dry run only. Re-run with --apply to write changes."
  exit 0
fi

# Literal replace via python: no regex, no delimiter escaping headaches.
replace_in_file() {
  python - "$1" "$2" "$3" <<'PY'
import sys, pathlib
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
if old == new:
    sys.exit(0)
p = pathlib.Path(path)
text = p.read_text()
count = text.count(old)
if count == 0:
    sys.exit(0)
p.write_text(text.replace(old, new))
print(f"  {path}: {count} replacement(s) of {old!r}")
PY
}

# Replace a KEY=<value> line in an env-style file regardless of the
# previous value. Returns silently if the key is absent.
replace_env_kv() {
  python - "$1" "$2" "$3" <<'PY'
import re, sys, pathlib
path, key, new_val = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(path)
text = p.read_text()
pat = re.compile(rf'^{re.escape(key)}=.*$', re.MULTILINE)
m = pat.search(text)
if not m:
    print(f"  {path}: no {key}= line; skipped")
    sys.exit(0)
old_line = m.group(0)
new_line = f"{key}={new_val}"
if old_line == new_line:
    sys.exit(0)
p.write_text(pat.sub(new_line, text, count=1))
print(f"  {path}: {old_line} -> {new_line}")
PY
}

echo "Applying replacements (doc/code files, literal match)..."
for f in "${FILES[@]}"; do
  if [ "$OLD_STRONG" != "$NEW_STRONG" ]; then
    replace_in_file "$f" "$OLD_STRONG"    "$NEW_STRONG"
    # Only replace basename if it differs from full id; otherwise the
    # full-id replace above already handled it.
    if [ "$OLD_STRONG_BN" != "$OLD_STRONG" ]; then
      replace_in_file "$f" "$OLD_STRONG_BN" "$NEW_STRONG_BN"
    fi
  fi
  if [ "$OLD_EVAL" != "$NEW_EVAL" ]; then
    replace_in_file "$f" "$OLD_EVAL"    "$NEW_EVAL"
    if [ "$OLD_EVAL_BN" != "$OLD_EVAL" ]; then
      replace_in_file "$f" "$OLD_EVAL_BN" "$NEW_EVAL_BN"
    fi
  fi
done
echo

echo "Applying replacements (env files, KEY=VALUE)..."
for f in "${ENV_FILES[@]}"; do
  replace_env_kv "$f" "WDOC_DEFAULT_MODEL"            "$NEW_STRONG"
  replace_env_kv "$f" "WDOC_DEFAULT_QUERY_EVAL_MODEL" "$NEW_EVAL"
done
echo

# Post-flight: scan target files for any lingering reference. The "family"
# check catches shortened aliases (e.g. "gemini-3.1-pro" in ARCHITECTURE.md
# vs the full "gemini-3.1-pro-preview" in env.py).
#
# Done in python to avoid a real issue when the old value is a substring of
# the new value (e.g. prepending "openrouter/" to an existing id): a naive
# grep for the old value would match inside the new value and report a
# false-positive leftover. We mask new-value spans before counting old.
echo "Post-flight scan:"
SUMMARY=$(python - \
  "$OLD_STRONG" "$NEW_STRONG" \
  "$OLD_STRONG_BN" "$NEW_STRONG_BN" \
  "$OLD_EVAL" "$NEW_EVAL" \
  "$OLD_EVAL_BN" "$NEW_EVAL_BN" \
  "$OLD_STRONG_FAM" "$OLD_EVAL_FAM" \
  "${FILES[@]}" "${ENV_FILES[@]}" <<'PY'
import sys, pathlib
args = sys.argv[1:]
(os_full, ns_full,
 os_bn,   ns_bn,
 oe_full, ne_full,
 oe_bn,   ne_bn,
 os_fam,  oe_fam) = args[:10]
files = args[10:]

# Dedupe pairs so basename == full-id doesn't double-report
pairs = []
seen = set()
for old, new in [(os_full, ns_full), (os_bn, ns_bn), (oe_full, ne_full), (oe_bn, ne_bn)]:
    if not old or old == new or (old, new) in seen:
        continue
    seen.add((old, new))
    pairs.append((old, new))

families = []
for fam in (os_fam, oe_fam):
    if fam and fam not in families:
        families.append(fam)

def true_leftover(text, old, new):
    """Return (line_no, line) for occurrences of old not contained in new."""
    if new:
        masked = text.replace(new, "\x01" * len(new))
    else:
        masked = text
    out = []
    for i, (mline, oline) in enumerate(zip(masked.splitlines(), text.splitlines()), 1):
        if old in mline:
            out.append((i, oline))
    return out

hard = 0
soft = 0
for path in files:
    text = pathlib.Path(path).read_text()
    for old, new in pairs:
        hits = true_leftover(text, old, new)
        if hits:
            print(f"  HARD: {old!r} still in {path}:")
            for i, line in hits:
                print(f"      {i}:{line}")
            hard += 1
    for fam in families:
        lower_fam = fam.lower()
        hits = [(i, line) for i, line in enumerate(text.splitlines(), 1)
                if lower_fam in line.lower()]
        if hits:
            print(f"  SOFT: family {fam!r} still mentioned in {path} (may be unrelated):")
            for i, line in hits:
                print(f"      {i}:{line}")
            soft += 1

print(f"__SUMMARY__ hard={hard} soft={soft}")
PY
)
echo "$SUMMARY" | sed '/^__SUMMARY__/d'
leftover_hard=$(echo "$SUMMARY" | sed -n 's/^__SUMMARY__ hard=\([0-9]\+\) soft=.*/\1/p')
leftover_soft=$(echo "$SUMMARY" | sed -n 's/^__SUMMARY__ .* soft=\([0-9]\+\).*/\1/p')

if [ "$leftover_hard" -eq 0 ] && [ "$leftover_soft" -eq 0 ]; then
  echo "  clean - no leftover references"
elif [ "$leftover_hard" -eq 0 ]; then
  echo
  echo "  No hard leftovers. $leftover_soft soft family mention(s) above;"
  echo "  check whether they refer to the bumped model or something else."
else
  echo
  echo "  $leftover_hard hard leftover(s) found - script missed them."
  echo "  Inspect manually and fix; the most likely cause is a short alias"
  echo "  whose basename differs from the env.py default (e.g. -preview suffix)."
fi

echo
echo "Done. Review with 'git diff' and commit when satisfied."
