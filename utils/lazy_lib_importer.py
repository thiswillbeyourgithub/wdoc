from pathlib import Path
import re
import lazy_import

pattern = re.compile(r'^from (?P<fsource>[\w\.]+) import (?P<fwhat>[\w,. ()]+)( as (?P<fas>\w+))?|^import (?P<iwhat>[\w()]+)( as (?P<ias>\w+))?$')


def lazy_import_statements(
    text: str,
    verbose: bool=False,
    bypass: bool=False,
    allow_local: bool=False,
    ) -> str:
    """turn a bunch of import statements into lazy imports
    it returns strings in the form of lines to execute with exec()
    For example "from A import B as Z" is turned into "Z = lazy_import.lazy_module(A.B)"
    The output lines are wrapped in a try block, and if fail, the original line is run
    This does not work will all import statements, and local imports throw an exception
    by default.
    Lazy loading does not seem to work with class import, for example when
    you plan to do an inheritance with it.
    """
    if bypass:
        return text
    assert isinstance(text, str)
    lines = text.splitlines()

    lines = [li.strip() for li in lines]
    orig_len = len(lines)
    il2 = 0
    for il, li in enumerate(lines):
        cnt = 0
        if li is None:
            continue
        while li.endswith("(") or li.endswith(","):
            cnt += 1
            for il2, li2 in enumerate(lines[il + 1:]):
                if li2 is not None:
                    break
            li = li + " " + li2
            lines[il] = li
            lines[il+il2 + 1] = None

            if il + il2 + 2 <= len(lines):
                if lines[il+il2+2].strip() == ")":
                    li = li + ")"
                    lines[il] = li
                    lines[il+il2+2] = None


            assert len(lines) == orig_len
            if cnt > 100:
                raise Exception("Infinite loop")

    lines = [li.strip() for li in lines if li]
    lines = [li.split("#")[0].strip() for li in lines]
    lines = [li for li in lines if li]

    statements = []
    lines2 = []

    for line in lines:
        assert line

        match = re.match(pattern, line)
        if not match:
            raise ValueError(f"No match: {line}")

        d = match.groupdict()
        fsource = d["fsource"] or ""
        fwhat = d["fwhat"] or ""
        fas = d["fas"] or ""
        iwhat = d["iwhat"] or ""
        ias = d["ias"] or ""

        fwhat = fwhat.replace("(", "").replace(")", "").strip()
        iwhat = iwhat.replace("(", "").replace(")", "").strip()

        fwhats = fwhat.split(", ") if ", " in fwhat else [fwhat]
        iwhats = iwhat.split(", ") if ", " in iwhat else [iwhat]

        if fsource:  # from X import Y (as Z)
            assert fwhat
            assert not iwhat
            assert not ias

            first_source = fsource.split(".")[0] if not fsource.startswith(".") else fsource.split(".")[1]
            if Path(first_source).exists():
                if allow_local:
                    statements.append(line.strip())
                    lines2.append(line)
                    continue
                else:
                    raise Exception(f"Detected local import: {line}")

            if ias:
                assert len(fwhats) == 1
                statements.append(f"{ias} = lazy_import.lazy_callable('{fsource}.{fwhat}')")
                lines2.append(line)
            else:
                for fwhat in fwhats:
                    statements.append(f"{fwhat} = lazy_import.lazy_callable('{fsource}.{fwhat}')")
                    lines2.append(line.split("import ")[0] + f"import {fwhat}")

        else:  # import X (as Y)
            assert not fwhat
            assert not fas
            assert iwhat

            if ias:
                assert len(iwhats) == 1
                statements.append(f"{ias} = lazy_import.lazy_module('{iwhat}')")
                lines2.append(line)
            else:
                for iwhat in iwhats:
                    assert "." not in iwhat
                    statements.append(f"{iwhat} = lazy_import.lazy_module('{iwhat}')")
                    lines2.append(line.split("import ")[0] + f"import {iwhat}")

    assert len(lines2) == len(statements)

    output = []
    for stat, line in zip(statements, lines2):
        output.append(f"""
try:
    {stat}
except Exception as err:
    print(f"Failed to lazyload '{stat}': '{err}'")
    {line}
""".strip()
                      )
    output = "\n".join(output)
    if verbose:
        print(output)
    return output
