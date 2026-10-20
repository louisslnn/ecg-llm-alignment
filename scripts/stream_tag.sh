# Sourced by the job scripts, not run. The one mapping from a --task-stream mode to
# the tag train.sh puts on a run's checkpoint dir (run2_question, ...), and that the
# gate, evaluation and probe scripts use to find that dir and name their outputs.
# One place, so a script cannot look for a directory under a name training never
# wrote, or file a run under a stream it did not read.
#
#   prompt -> ""   prompt-and-target -> _withtarget   question -> _question
#   none -> _notext
#
# stream_tag MODE prints the tag, and fails (status 1, nothing printed) on an
# unknown mode, so a caller can say so before the module load:
#   STREAM_TAG="$(stream_tag "$TASK_STREAM")" || { echo "..." >&2; exit 2; }

STREAM_MODES="prompt, prompt-and-target, question or none"

stream_tag() {
    case "$1" in
        prompt)            echo "" ;;
        prompt-and-target) echo "_withtarget" ;;
        question)          echo "_question" ;;
        none)              echo "_notext" ;;
        *)                 return 1 ;;
    esac
}
