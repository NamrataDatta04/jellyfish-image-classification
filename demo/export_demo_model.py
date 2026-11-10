"""
export_demo_model.py  -  builds the files used by the in-browser demo (docs/index.html)
=====================================================================================
Needs only numpy, onnx, onnxruntime, Pillow and scikit-learn (no PyTorch, no GPU).

1. Reads a trained torchvision EfficientNet-B0 checkpoint (.pt) directly from its zip/pickle format.
2. Writes the network WITHOUT its last layer as an ONNX graph (batch-norm folded into the convolutions),
   so the browser gets the 1280 penultimate features.
3. Saves the model's own last layer ("standard training", ERM) as plain weights.
4. Re-fits the DFR last layer with the same recipe as dfr.py / extra_common.fit_group_balanced_head
   (group-balanced logistic regression on the validation groups of the class x background benchmark,
   C chosen by grouped 4-fold CV on worst-group accuracy) and folds the feature scaler into it.
5. Prints the accuracy of both heads on the benchmark test groups, to compare with
   results_snapshot/extra/all_methods.csv.

Run from the repository root (needs the benchmark crops written by group_benchmark.py):
    python demo/export_demo_model.py --ckpt outputs/checkpoints/binary_efficientnet_b0_pt_ft_aug-standard_s813.pt \
                                     --groups_csv outputs/groups.csv --outputs_dir outputs
Writes docs/model/features.onnx and docs/model/heads.json.
"""

import argparse
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BN_EPS = 1e-5
DFR_GROUPS = ["bag_rov", "bag_gray", "jelly_clear", "jelly_gray", "jelly_rov", "jelly_insitu"]
CORE_GROUPS = ["bag_rov", "bag_clear", "jelly_clear", "jelly_rov"]   # worst-group accuracy is over these 4
SHOW_GROUPS = CORE_GROUPS + ["bag_gray", "jelly_gray", "jelly_insitu"]
SEED = 813


# ---------------------------------------------------------------------------------------------
# 1. Read a PyTorch checkpoint without PyTorch
# ---------------------------------------------------------------------------------------------
_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "LongStorage": np.int64,
           "IntStorage": np.int32, "BoolStorage": np.bool_, "DoubleStorage": np.float64}


def load_pt(path):
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split("/")[0]

    def rebuild_tensor(storage, offset, size, stride, *_):
        if len(size) == 0:
            return storage[offset].copy()
        itemsize = storage.itemsize
        return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                               strides=tuple(s * itemsize for s in stride)).copy()

    class _Storage:
        def __init__(self, name):
            self.name = name

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild_tensor
            if module == "torch" and name in _DTYPES:
                return _Storage(name)
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if module.startswith("torch") or module.startswith("numpy"):
                return lambda *a, **k: None          # anything else (e.g. torch.device) is not needed
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, storage_type, key, _, _ = pid
            dtype = _DTYPES[storage_type.name]
            return np.frombuffer(zf.read(f"{prefix}/data/{key}"), dtype=dtype)

    return Unpickler(zf.open(f"{prefix}/data.pkl")).load()


# ---------------------------------------------------------------------------------------------
# 2. EfficientNet-B0 feature extractor as an ONNX graph (torchvision layout, eval mode)
# ---------------------------------------------------------------------------------------------
class GraphBuilder:
    def __init__(self, sd):
        self.sd, self.nodes, self.inits, self.n = sd, [], [], 0

    def name(self, base):
        self.n += 1
        return f"{base}_{self.n}"

    def const(self, arr, base):
        nm = self.name(base)
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), nm))
        return nm

    def op(self, op_type, inputs, **attrs):
        out = self.name(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def conv_bn(self, x, key, stride=1, groups=1, act=True):
        """Conv2d (no bias) + BatchNorm folded into one Conv, then optional SiLU."""
        w = self.sd[f"{key}.0.weight"].astype(np.float32)
        g, b = self.sd[f"{key}.1.weight"], self.sd[f"{key}.1.bias"]
        mu, var = self.sd[f"{key}.1.running_mean"], self.sd[f"{key}.1.running_var"]
        scale = g / np.sqrt(var + BN_EPS)
        w_f = (w * scale[:, None, None, None]).astype(np.float32)
        b_f = (b - mu * scale).astype(np.float32)
        k = w.shape[-1]
        y = self.op("Conv", [x, self.const(w_f, "w"), self.const(b_f, "b")], kernel_shape=[k, k],
                    strides=[stride, stride], pads=[(k - 1) // 2] * 4, group=groups)
        return self.silu(y) if act else y

    def silu(self, x):
        return self.op("Mul", [x, self.op("Sigmoid", [x])])

    def squeeze_excite(self, x, key):
        s = self.op("GlobalAveragePool", [x])
        s = self.op("Conv", [s, self.const(self.sd[f"{key}.fc1.weight"].astype(np.float32), "w"),
                             self.const(self.sd[f"{key}.fc1.bias"].astype(np.float32), "b")], kernel_shape=[1, 1])
        s = self.silu(s)
        s = self.op("Conv", [s, self.const(self.sd[f"{key}.fc2.weight"].astype(np.float32), "w"),
                             self.const(self.sd[f"{key}.fc2.bias"].astype(np.float32), "b")], kernel_shape=[1, 1])
        return self.op("Mul", [x, self.op("Sigmoid", [s])])


# torchvision efficientnet_b0: (expand ratio, kernel, stride, in, out, layers) per stage
B0_STAGES = [(1, 3, 1, 32, 16, 1), (6, 3, 2, 16, 24, 2), (6, 5, 2, 24, 40, 2), (6, 3, 2, 40, 80, 3),
             (6, 5, 1, 80, 112, 3), (6, 5, 2, 112, 192, 4), (6, 3, 1, 192, 320, 1)]


def build_efficientnet_b0_features(sd):
    gb = GraphBuilder(sd)
    x = gb.conv_bn("image", "features.0", stride=2)
    for s, (expand, k, stride, cin, cout, layers) in enumerate(B0_STAGES, start=1):
        for i in range(layers):
            key = f"features.{s}.{i}.block"
            inp, c_in, st = x, (cin if i == 0 else cout), (stride if i == 0 else 1)
            j = 0
            if expand != 1:
                x = gb.conv_bn(x, f"{key}.0"); j = 1
            hidden = c_in * expand
            x = gb.conv_bn(x, f"{key}.{j}", stride=st, groups=hidden)
            x = gb.squeeze_excite(x, f"{key}.{j + 1}")
            x = gb.conv_bn(x, f"{key}.{j + 2}", act=False)
            if st == 1 and c_in == cout:                # residual (stochastic depth = identity at eval)
                x = gb.op("Add", [x, inp])
    x = gb.conv_bn(x, "features.8")
    x = gb.op("GlobalAveragePool", [x])
    gb.nodes.append(helper.make_node("Flatten", [x], ["features"], axis=1))
    graph = helper.make_graph(
        gb.nodes, "efficientnet_b0_features",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["n", 3, IMG_SIZE, IMG_SIZE])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", 1280])], gb.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="export_demo_model")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# ---------------------------------------------------------------------------------------------
# 3. Images and the DFR head (same as extra_common.load_uint8 / to_model_input / fit_group_balanced_head)
# ---------------------------------------------------------------------------------------------
def load_batch(paths):
    arr = np.stack([np.asarray(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
                    for p in paths]).astype(np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)


def extract(sess, paths, batch=32):
    return np.concatenate([sess.run(None, {"image": load_batch(paths[i:i + batch])})[0]
                           for i in range(0, len(paths), batch)])


def fit_group_balanced_head(X, y, groups, clusters, Cs=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    groups = np.asarray(groups)
    gw = pd.Series(groups).map(1.0 / pd.Series(groups).value_counts()).values
    gw = gw / gw.mean()
    best = (-1.0, Cs[0])
    for c in Cs:
        accs = []
        for tr, va in GroupKFold(n_splits=min(4, len(np.unique(clusters)))).split(X, y, clusters):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(X[tr]), y[tr], sample_weight=gw[tr])
            pred = clf.predict(sc.transform(X[va]))
            accs.append(min(np.mean(pred[groups[va] == g] == y[va][groups[va] == g]) for g in np.unique(groups[va])))
        score = float(np.mean(accs)) if accs else -1
        if score > best[0]:
            best = (score, c)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=best[1], max_iter=5000).fit(sc.transform(X), y, sample_weight=gw)
    return sc, clf, best[1]


def remap(path, outputs_dir):
    """groups.csv stores absolute paths from the training machine; point them at this machine's outputs folder."""
    p = str(path).replace("\\", "/")
    i = p.lower().find("/outputs/")
    return str(Path(outputs_dir) / p[i + len("/outputs/"):]) if i >= 0 else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--groups_csv", required=True)
    ap.add_argument("--outputs_dir", required=True, help="folder that holds crops/ (remaps the paths in groups.csv)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "model"))
    args = ap.parse_args()
    np.random.seed(SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = load_pt(args.ckpt)
    assert ck["model"] == "efficientnet_b0" and ck["task"] == "binary", "the demo expects a binary EfficientNet-B0"
    sd = ck["state_dict"]
    onnx_path = out / "features.onnx"
    onnx.save(build_efficientnet_b0_features(sd), onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    w_erm, b_erm = sd["classifier.1.weight"].astype(float), sd["classifier.1.bias"].astype(float)

    g = pd.read_csv(args.groups_csv)
    g["path"] = [remap(p, args.outputs_dir) for p in g.path]
    rw = g[(g.set == "gval") & g.group.isin(DFR_GROUPS) & (g.y_bin >= 0)].reset_index(drop=True)
    te = g[(g.set == "gtest") & g.group.isin(SHOW_GROUPS)].reset_index(drop=True)
    print(f"DFR fitting set: {len(rw)} images | benchmark test groups: {len(te)} images")
    sc, clf, c_best = fit_group_balanced_head(extract(sess, rw.path.tolist()), rw.y_bin.values.astype(int),
                                              rw.group.values, rw.cluster.values)
    assert list(clf.classes_) == [0, 1]
    w_dfr = clf.coef_[0] / sc.scale_                                   # scaler folded into the weights
    b_dfr = float(clf.intercept_[0] - np.sum(clf.coef_[0] * sc.mean_ / sc.scale_))

    f_te = extract(sess, te.path.tolist())
    erm_pred = (f_te @ w_erm.T + b_erm).argmax(1)
    dfr_pred = ((f_te @ w_dfr + b_dfr) > 0).astype(int)
    rows = []
    for grp in SHOW_GROUPS:
        m = (te.group == grp).values
        rows.append({"group": grp, "n": int(m.sum()),
                     "standard": round(float((erm_pred[m] == te.y_bin.values[m]).mean()), 4),
                     "dfr": round(float((dfr_pred[m] == te.y_bin.values[m]).mean()), 4)})
    acc = pd.DataFrame(rows)
    print(acc.to_string(index=False))
    core = acc[acc.group.isin(CORE_GROUPS)]
    print(f"worst-group accuracy: standard {core.standard.min():.3f} | DFR {core.dfr.min():.3f} (C={c_best})")

    heads = {
        "model": "EfficientNet-B0, ImageNet-pretrained, fine-tuned on the binary task (seed 813)",
        "checkpoint": Path(args.ckpt).name,
        "classes": ["plastic bag", "jellyfish"],
        "input": {"size": IMG_SIZE, "mean": MEAN.tolist(), "std": STD.tolist(), "resize": "squash to 224x224"},
        "erm": {"weight": np.round(w_erm, 7).tolist(), "bias": np.round(b_erm, 7).tolist()},
        "dfr": {"weight": np.round(w_dfr, 7).tolist(), "bias": round(b_dfr, 7), "C": c_best},
        "benchmark_test_accuracy": rows,
    }
    (out / "heads.json").write_text(json.dumps(heads))
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) and {out / 'heads.json'}")


if __name__ == "__main__":
    main()
