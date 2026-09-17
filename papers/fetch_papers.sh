#!/usr/bin/env bash
# Download the public reference papers into papers/ and check them against the
# checksums recorded in papers/README.md. Run from anywhere.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ua="Mozilla/5.0 (pdes-demo fetch script)"

# name|url|sha256
papers=(
  "openai-2026-finite-time-blowup-navier-stokes.pdf|https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf|0e779481c4da40bd28d1e642e1d8ca57447d129610df28dfa5a11e9af8ae228f"
  "clay-2000-fefferman-navier-stokes-problem-statement.pdf|https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf|c1b5f27b1a64705cfaf1afceea513db5deedca8a18ca56ab32e7f86445a06d0c"
  "martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf|https://www.colorado.edu/amath/sites/default/files/attached-files/2015_mfstc_rbf-fd_2d_geophys_submitted_0.pdf|4f4cee20bf0683e15c99c8776c6c8422b3840ac17f8ae517a46a73a7e66c8f51"
  "martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf|https://www.colorado.edu/amath/sites/default/files/attached-files/2016_mf_rbf-fd_seismic_jcp_submitted.pdf|dd862ff7e69830e5bdd8ee72abccfd0bbdb96c3a8374df8a3733af25c86b1a58"
)

for entry in "${papers[@]}"; do
  IFS='|' read -r name url sha <<<"$entry"
  dest="$here/$name"
  if [[ -f "$dest" ]]; then
    echo "exists  $name"
  else
    echo "fetch   $name"
    curl -fsSL -A "$ua" -o "$dest" "$url"
  fi
  got="$(shasum -a 256 "$dest" | cut -d' ' -f1)"
  if [[ "$got" == "$sha" ]]; then
    echo "ok      $name"
  else
    echo "WARNING $name: sha256 $got != recorded $sha (upstream revised?)" >&2
  fi
done

diss="$here/martin-dissertation-2016-rbf-fd-interfaces.pdf"
diss_sha="a658c8b94547eb085eed68e1bd70cca00ed9c49eba8911d8412576d9945e96f2"
if [[ -f "$diss" ]]; then
  got="$(shasum -a 256 "$diss" | cut -d' ' -f1)"
  if [[ "$got" == "$diss_sha" ]]; then
    echo "ok      $(basename "$diss") (hand-supplied)"
  else
    echo "WARNING $(basename "$diss"): sha256 differs from the copy indexed 2026-09-17" >&2
  fi
else
  echo "note    dissertation PDF not present; see papers/README.md" >&2
fi
