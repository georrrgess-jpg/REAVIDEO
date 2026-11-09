#!/usr/bin/env bash
# Re-download the stored scene packs and references from Google Drive and verify them.
#   bash scenepacks/fetch.sh            # everything (~800 MB)
#   bash scenepacks/fetch.sh dexter     # only files whose name contains "dexter"
# Files land in media/scenepacks/ and media/references/ (gitignored). Existing files that
# already match their checksum are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
filter="${1:-}"; fail=0
while IFS=$'\t' read -r kind file id bytes sha w h fps secs title; do
  [[ "$kind" == \#* ]] && continue
  [[ -n "$filter" && "$file" != *"$filter"* ]] && continue
  dir="media/$([ "$kind" = reference ] && echo references || echo scenepacks)"; mkdir -p "$dir"; out="$dir/$file"
  if [ -f "$out" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; then echo "ok (cached)  $out"; continue; fi
  for try in 1 2 3 4; do
    curl -sSfL -o "$out.part" "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t" && break
    sleep $((2 ** try))
  done
  if [ -f "$out.part" ] && [ "$(sha256sum "$out.part" | cut -d' ' -f1)" = "$sha" ]; then mv "$out.part" "$out"; echo "ok           $out"
  else rm -f "$out.part"; echo "FAILED       $out (download or checksum; the Drive link may have changed)" >&2; fail=1; fi
done < scenepacks/manifest.tsv
exit $fail
