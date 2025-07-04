from pathlib import Path


for file in Path().glob("*.cu?"):
    result = []
    last_was_pragma = None
    for line in open(file):
        if last_was_pragma:
            loop_at = line.find("for")
            result.append(" " * loop_at + last_was_pragma + line)
            last_was_pragma = False
            continue
        if line.startswith("#pragma unroll"):
            last_was_pragma = line
        else:
            result.append(line)
    file.write_text(str.join("", result))
