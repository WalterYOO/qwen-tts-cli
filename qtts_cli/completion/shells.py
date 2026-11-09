"""Deterministic shell adapter generation.

The adapters are intentionally small: all command/option knowledge lives in
``qtts __complete``.  This keeps bash/zsh/fish from drifting away from the
argparse tree and avoids shipping a second copy of the CLI grammar.
"""

from __future__ import annotations

from ..errors import UsageError
from .detect import SUPPORTED_SHELLS

_BASH = """\
# qtts completion for bash
_qtts_completion() {
    local out kind line
    COMPREPLY=()
    out=$(command "${QTTS_COMPLETION_BIN:-qtts}" __complete \\
        --shell bash --line "$COMP_LINE" --point "$COMP_POINT" 2>/dev/null) || return 0
    [ -n "$out" ] || return 0
    kind=${out%%$'\\n'*}
    if [[ "$out" == *$'\\n'* ]]; then
        out=${out#*$'\\n'}
    else
        out=""
    fi
    case "$kind" in
        *kind=file*|*kind=dir*) return 0 ;;
    esac
    while IFS= read -r line; do
        [ -n "$line" ] && COMPREPLY+=("$line")
    done <<< "$out"
    return 0
}
complete -o filenames -o bashdefault -o default -F _qtts_completion qtts
"""

_ZSH = """\
# qtts completion for zsh
_qtts_completion() {
    local -a out candidates argv
    local kind
    argv=("${words[@]}")
    (( CURRENT > ${#words} )) && argv+=("")
    out=("${(@f)$(command "${QTTS_COMPLETION_BIN:-qtts}" __complete \\
        --shell zsh --cword $((CURRENT - 1)) -- "${argv[@]}" 2>/dev/null)}")
    (( ${#out} )) || return 0
    kind=${out[1]}
    candidates=("${(@)out[2,-1]}")
    case "$kind" in
        *kind=file*) _files ;;
        *kind=dir*) _directories ;;
        *) (( ${#candidates} )) && compadd -- "${candidates[@]}" ;;
    esac
}
if (( $+functions[compdef] )); then
    compdef _qtts_completion qtts
else
    autoload -Uz compinit && compinit
    compdef _qtts_completion qtts
fi
"""

_FISH = """\
# qtts completion for fish
function __qtts_complete
    set -l tokens (commandline -opc)
    set -l current (commandline -ct)
    set -l bin qtts
    if set -q QTTS_COMPLETION_BIN
        set bin $QTTS_COMPLETION_BIN
    end
    set -l out (command $bin __complete --shell fish \\
        --cword (count $tokens) -- $tokens "$current" 2>/dev/null)
    test (count $out) -gt 0; or return 0
    set -l kind $out[1]
    if string match -q '*kind=file*' -- $kind
        __fish_complete_path $current
    else if string match -q '*kind=dir*' -- $kind
        __fish_complete_directories $current
    else if test (count $out) -gt 1
        for candidate in $out[2..-1]
            printf '%s\\n' $candidate
        end
    end
end

complete -c qtts -a '(__qtts_complete)'
"""

_SCRIPTS = {"bash": _BASH, "zsh": _ZSH, "fish": _FISH}


def generate_script(shell: str) -> str:
    """Return the adapter for ``shell`` with exactly one trailing newline."""
    if shell not in SUPPORTED_SHELLS:
        raise UsageError(f"unsupported shell {shell!r}; supported shells: bash, zsh, fish")
    script = _SCRIPTS[shell]
    if not script.endswith("\n"):
        script += "\n"
    return script
