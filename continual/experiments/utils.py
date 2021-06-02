import json
import ast
import pandas as pd

def cleaned_up_json(path):
    """Cleans up json logs of a given experiment"""
    out = {}
    def clean(st, loss=False):
        if not loss:
            st = st.replace('\n', "")
            st = st.replace('   ', ",")
            st = st.replace('  ', ",")
            st = st.replace(' ', ",")
            st = st.replace('[,', "[")
            st = st.replace(',]', "]")
        st = ast.literal_eval(st)
        st = pd.DataFrame(st)
        return st


    with open(path) as f:
        s = json.load(f)
        out["name"] = s["name"].split("/")[-2]
        out["avg_acc"] = s["avg_acc"]
        out["gem_bwt"] = s["gem_bwt"]
        out["ucb_bwt"] = s["ucb_bwt"]
        out["taskcla"] = s["taskcla"]
        out["acc"] = clean(s["acc"])
        out["loss"] = clean(s["loss"], True)
        out["rii"] = clean(s["rii"])
        out["rij"] = clean(s["rij"])
        out["labels_per_task"] = clean(s["labels_per_task"], True)
    return out

def get_final_acc(path):
    path = path + "/Final_results.json"
    with open(path) as f:
        s = json.load(f)
        return ast.literal_eval(s["avg_acc"])

def bg_color(val):
    if val == 0.0:
        color = 'black'
    elif val <= 20:
        color = 'red'
    elif val < 30:
        color = '#ff6700'
    elif val < 40:
        color = '#ffb400'
    elif val < 50:
        color = '#ffdb00'
    elif val < 60:
        color = '#e3ff00'
    elif val < 70:
        color = '#96ff00'
    elif val < 80:
        color = '#23bf00'
    elif val <= 100:
        color = 'green'
    return f'background-color: {color}'

def color(val):
    if val > 0.0:
        color = 'black'
    else:
        color = 'white'
    return f'color: {color}'
