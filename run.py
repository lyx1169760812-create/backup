import os
# 71  蝎子   75 蜘蛛
# 84  孔雀   97 鸭子
# 121 帝王蟹 122 龙虾
# 150  海狮  205 拉布拉多
# 250  二哈 281 猫
# 291 292 狮子 老虎
# 387 小熊猫 388 大熊猫
# 344 河马 346 水牛
# 9   鸵鸟  22 鹰
# 34  棱皮龟 38 蜥蜴 
merge_class = ['[387,279]', '[291,292]', '[344,346]', '[9,22]', '[34,38]','[71,75]','[84,97]','[121,122]','[150,205]','[250,281]']
# for merges in merge_class:
#     for i in range(20):
#         cmd = 'python sample.py --image-size 512 --merge-cls ' + merges + ' --out-img-num 1 --num-sampling-steps 200 --weight-strategy 1 --seed ' + str(i)
#         print(cmd)
        # os.system(cmd)
for merges in merge_class:
    for i in range(20):
        cmd = 'python sample.py --image-size 512 --merge-cls ' + merges + ' --out-img-num 1 --num-sampling-steps 200 --weight-strategy 0 --seed ' + str(i)
        print(cmd)
        os.system(cmd)