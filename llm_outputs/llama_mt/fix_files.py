def fix_files():
    import os
    fns = os.listdir('.')
    for fn in fns:
        cmd = f"mv {fn} {fn.replace('-', '_to_').replace('BTEC', 'Madar26').replace("FLORES", "Flores200")}"
        os.system(cmd)
        print(cmd)

if __name__ == "__main__":
    fix_files() 