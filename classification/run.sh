#for MODEL in resnet18 resnet50 inceptionv3 mobilenetv2_w1 shufflenet_g1_w1 sqnxt23_w2
for MODEL in resnet18
do
	echo Testing $MODEL ...
  CUDA_VISIBLE_DEVICES=7 \
	python uniform_test.py 		\
		--dataset=imagenet 		\
		--model=$MODEL 			\
		--batch_size=64 		\
		--test_batch_size=512
done
