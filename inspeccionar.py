import onnxruntime as ort
sess = ort.InferenceSession('best.onnx')
print('ENTRADAS:')
for i in sess.get_inputs():
    print(' ', i.name, i.shape)
print('SALIDAS:')
for o in sess.get_outputs():
    print(' ', o.name, o.shape)