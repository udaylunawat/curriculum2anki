import re
import sys

def check_latex(fname):
    begin = r"\[\$\]"
    end = r"\[\/\$\]"
    pattern = re.compile(begin + "(.+?)" + end, re.DOTALL)
    failures = []
    with open(fname) as f:
        for i,line in enumerate(f,1):
            for block in pattern.findall(line):
                # crude check for balanced braces
                if block.count("{") != block.count("}"):
                    failures.append((i,block))
    if failures:
        for ln, blk in failures:
            print(f"Unbalanced LaTeX line {ln}: {blk}")
    else:
        print("Basic LaTeX validation OK")

if __name__ == "__main__":
    check_latex(sys.argv[1])