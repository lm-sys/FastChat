#!/usr/bin/env bash

# Adopted from https://github.com/skypilot-org/skypilot/blob/master/format.sh

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

BLACK_VERSION=$(black --version | head -n 1 | awk '{print $2}')
PYLINT_VERSION=$(pylint --version | head -n 1 | awk '{print $2}')

# # params: tool name, tool version, required version
tool_version_check() {
    if [[ $2 != $3 ]]; then
        echo "Wrong $1 version installed: $3 is required, not $2."
        exit 1
    fi
}

tool_version_check "black" $BLACK_VERSION "23.3.0"
tool_version_check "pylint" $PYLINT_VERSION "2.8.2"

# Format files that differ from main branch. Ignores dirs that are not slated
# for autoformat yet.
format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause yapf to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only format files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 black
    fi
}

## This flag formats individual files. --files *must* be the first command line
## arg to use this option.
if [[ "$1" == '--files' ]]; then
   black "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is formatted.
elif [[ "$1" == '--all' ]]; then
   # Format all files
   black fastchat
else
   # Format only the files that changed in last commit.
   format_changed
fi
echo 'FastChat Black: Done'

# Run Pylint
echo 'FastChat Pylint:'
pylint fastchat
# TODO(suquark): disable 'pylint_quotes' for now due to too many inconsistent quotes
# pylint --load-plugins pylint_quotes fastchat

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi
