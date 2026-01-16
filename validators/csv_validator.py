import csv
import sys

def validate_csv(fname):
    with open(fname) as f:
        reader = csv.reader(f)
        errors = 0
        for i,row in enumerate(reader,1):
            if len(row) != 3:
                print(f"Line {i} wrong column count: {row}")
                errors+=1
        if not errors:
            print("CSV basic structure OK")

if __name__ == "__main__":
    validate_csv(sys.argv[1])