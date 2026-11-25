"""Run the official test and check(entry_point) in a sandboxed subprocess."""
import contextlib
import json
import re
import resource
import sys
import textwrap


def grade(request):
    text = request["text"].strip("\n")
    fenced = re.search(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    program = textwrap.dedent(text) + "\n\n" + request["test"]
    namespace = {}
    with open("/dev/null", "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(program, "<humaneval>", "exec"), namespace)
        namespace["check"](namespace[request["entry_point"]])
    return "pass"


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    try:
        result = grade(json.load(sys.stdin))
    except BaseException as error:
        result = type(error).__name__
    print(result)
