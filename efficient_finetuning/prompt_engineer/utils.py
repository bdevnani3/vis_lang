import yaml

def parse_yaml(path):
    f = open(path, 'r')
    out = yaml.load(f)
    f.close()
    return out


def generate_config_string(d):
    out = ""
    for k,v in d.items():
        if len(out)>0:
            out+=("_")
        k_str = str(k).replace("_","-")
        out+=(str(k_str))
        out+=(str(v))
    return out