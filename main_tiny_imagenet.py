from test_tiny_imagenet import fgsm_uap_test
from test_tiny_imagenet import fgsm_one_uap_test
from test_tiny_imagenet import fgsm_class_uap_test

fgsm_uap_test(seed=0)
fgsm_uap_test(seed=1)
fgsm_uap_test(seed=2)

# fgsm_one_uap_test(seed=0)
# fgsm_one_uap_test(seed=1)
# fgsm_one_uap_test(seed=2)

# fgsm_class_uap_test(seed=0)
# fgsm_class_uap_test(seed=1)
# fgsm_class_uap_test(seed=2)
