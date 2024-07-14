def build_cmd(script, param_dict):
    return f"python {script}.py" + "".join(f" --{k}={v}" for k, v in param_dict.items())
