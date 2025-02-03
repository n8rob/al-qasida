def fix_files():
    import os
    fns = os.listdir('.')
    fns = [fn for fn in fns if not fn.endswith('.py')]
    for fn in fns:
        assert fn.count('-') == 1 or fn.count('_to_') == 1
        new_fn = fn.replace('-', '_to_').replace('BTEC', 'Madar26').replace('FLORES', 'Flores200')
        if new_fn != fn:
            cmd = f"mv {fn} {new_fn}"
            os.system(cmd)
            print(cmd)

if __name__ == "__main__":
    fix_files() 