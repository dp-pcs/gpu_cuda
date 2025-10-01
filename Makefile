verify:
	python -c "import torch;print('cuda',torch.cuda.is_available());import platform;print(platform.platform())"

train-cpu:
	python src/train.py --device cpu --epochs 1 --batch-size 128

train-cuda:
	python src/train.py --device cuda --epochs 1 --batch-size 128

train-mps:
	python src/train.py --device mps --epochs 1 --batch-size 128

export-onnx:
	python src/export_onnx.py --out model.onnx

trt-benchmark:
	python src/infer_trt.py --onnx model.onnx --batch-size 1,8,32
