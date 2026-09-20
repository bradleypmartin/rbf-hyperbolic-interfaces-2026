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
  "openai-2026-finite-time-blowup-euler.pdf|https://cdn.openai.com/pdf/315b36cd-ec98-4023-8342-93345194ece1/euler.pdf|a0c234518e6c489e16996805023eb2e75c00b7c03455f7a3a5be2c124954bfdd"
  "alpoge-buckmaster-2026-euler-blowup-smooth-forcing.pdf|https://cims.nyu.edu/~tristanb/euler.pdf|97ef408bff09b4f6ed9f3867734d1eb2245f3f34e6334b28136c84c02d0ae8d8"
  "buckmaster-2026-statement.pdf|https://cims.nyu.edu/~tristanb/statement.pdf|8d7723941bcda2fa55c1e74faa6298e04c706d17ff8abd2ad01878039c621f9d"
  "tornberg-engquist-2006-regularization-wave-propagation-maa.pdf|https://www.intlpress.com/site/pub/files/_fulltext/journals/maa/2006/0013/0003/MAA-2006-0013-0003-a003.pdf|fc4d63910bd79bc506d5ac21382ccc707033cc3e4f27e31a784dbd39f4a1185f"
  "koene-wittsten-robertsson-2021-anti-aliasing-vs-equivalent-medium-arxiv.pdf|https://arxiv.org/pdf/2104.08206v2|5a52ef3473ae16a1276712813179a11df9e6d163580ffa89430d50a1a66cf826"
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
