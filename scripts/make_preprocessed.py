from argparse import ArgumentParser

argument_parser = ArgumentParser()
argument_parser.add_argument("-i", "--input_file")
argument_parser.add_argument("-o", "--out_file")

if __name__=="__main__":
    args = argument_parser.parse_args()
    lines = []
    with open(args.input_file, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    with open(args.out_file, "w", encoding="utf8") as fin:
        for line in lines:
            tokens = line.split()
            print(f"<CLS>\tKeep>Keep", file=fin)
            for token in tokens:
                print(f"{token}\tKeep>Keep", file=fin)
            print("", file=fin)   
    