# Store every float32 initializer as float16 with a Cast back to float32, so the
# download halves while all arithmetic stays in float32 (CPU and WebGPU alike).
import onnx, numpy as np, sys
from onnx import helper, numpy_helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
new_inits, casts = [], []
for init in g.initializer:
    a = numpy_helper.to_array(init)
    if a.dtype == np.float32 and a.size > 16:
        h = numpy_helper.from_array(a.astype(np.float16), init.name + '_fp16')
        new_inits.append(h)
        casts.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + '_cast'))
    else:
        new_inits.append(init)
del g.initializer[:]
g.initializer.extend(new_inits)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.checker.check_model(m)
onnx.save(m, dst)
print('saved', dst)
