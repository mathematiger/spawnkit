#!/usr/bin/env bash
# Run the test suite the way CI runs it, and make a failure readable without downloading the log.
#
# Both CI jobs call this — the torch-free interpreter matrix and the CPU-torch service tier — so the
# reporting cannot drift between them and neither can fail silently. The service tier used to: it
# exited 2 with nothing but "Process completed with exit code 2" in the checks API.
#
# A failure is mirrored into the job summary and re-emitted as workflow error commands. Annotations
# are readable through the public checks API, so the reason reaches anyone looking at the repository
# without a token to download the raw log. The reporting itself never changes the exit status: this
# script exits with pytest's, and nothing else.
#
# Usage: scripts/ci_pytest.sh <label>   (the label names the interpreter in the annotations)

set -uo pipefail

label="${1:?usage: ci_pytest.sh <label>}"
log="pytest.log"
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"

pytest -q -m "not gpu" -rf --timeout=300 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

if [ "$status" -eq 0 ]; then
    exit 0
fi

if [ ! -s "$log" ]; then
    echo "::error title=pytest ${label}::pytest exited ${status} without producing any output"
    exit "$status"
fi

{
    echo "### pytest failed on ${label} (exit ${status})"
    echo '```'
    tail -n 200 "$log"
    echo '```'
} >>"$summary"

# The named failures first, then the tail, which is what carries a collection error - that has no
# FAILED line at all, only a traceback, and it is how this suite most often breaks in CI.
grep -E "^(FAILED|ERROR)" "$log" | head -n 20 | while IFS= read -r line; do
    echo "::error title=pytest ${label}::${line}"
done
tail -n 25 "$log" | while IFS= read -r line; do
    echo "::error title=tail ${label}::${line}"
done

exit "$status"
