import math
import numpy as np
import pandas as pd

class Gentic_RF:

    def __init__(self, data, rf_model):
        self.rf_model = rf_model
        self.data = data
        self.original_data = np.array(data)
        self.data = np.array(self.data)

    def random_variation(self, rand_data, gen_num, v_range):
        """
        args:
            rand_data:要进行变异的随机数据数量
            gen_num:每次变异的基因数量
            range:数值变异范围
        """

        data_total = self.data.shape[0]
        gen_total = self.data.shape[1]

        sample_data = self.data

        if rand_data > data_total:
            print("变异数据量大于原数据量")
            return -1
        if gen_num > gen_total:
            print("变异基因量大于原基因量")
            return -1

        # 选择随机的数据进行变异
        sample = np.random.choice(np.arange(data_total), size=rand_data, replace=False)
        rand_gen_index = np.random.randint(low=0, high=gen_total, size=gen_num)

        tmp = sample_data[sample, 1]*np.random.uniform(1-v_range, 1+v_range)

        for i in rand_gen_index:
            sample_data[sample, i] = sample_data[sample, i]*np.random.uniform(1-v_range, 1+v_range)
            # 将数据剪枝至0.001~0.999范围内
            # self.data = np.clip(self.data, 0.001, 0.999)

        return sample_data[sample]

    def random_Genetic(self, rand_data, gen_num):
        """
        args:
            rand_data:要进行遗传的随机数据数量
            gen_num:每次遗传替换的基因数量
        """

        data_total = self.data.shape[0]
        gen_total = self.data.shape[1]

        sample_data = self.data

        if rand_data*2 > data_total:
            print("采样数据量大于原数据量")
            return -1
        if gen_num > gen_total:
            print("采样基因量大于原基因量")
            return -1

        sample = np.random.choice(np.arange(data_total), size=rand_data*2, replace=False)

        sample_child_gen = sample[:rand_data]
        sample_parent_gen = sample[rand_data:]

        rand_gen_index = np.random.randint(low=0, high=gen_total, size=gen_num)

        for i in rand_gen_index:
            sample_data[sample_child_gen, i] = sample_data[sample_parent_gen, i]

        return sample_data[sample_child_gen]

    def estimate(self, new_data, test_data=None, test_label=None):

        res = self.rf_model.predict(new_data)
        return res

    def fit_one_gen_iter(self, rand_data, gen_num, v_range, sort, test_data=None, test_label=None):
        """"
        单次遗传迭代
        args:
            rand_data:要进行变异的随机数据数量
            gen_num:每次变异的基因数量
            v_range:数值变异范围
            sort:+1为取低值，-1为取高值
        """

        data_total = self.data.shape[0]
        gen_total = self.data.shape[1]

        g_data = self.random_Genetic(rand_data, gen_num)
        v_data = self.random_variation(rand_data, gen_num, v_range)
        new_data = np.row_stack((v_data, g_data))
        new_data = np.row_stack((self.data, new_data))

        np.random.shuffle(new_data)

        res = {}
        all_res = 0

        this_res = self.estimate(new_data)
        new_data_label = np.c_[new_data, this_res]
        new_data_label = new_data_label[np.argsort(sort*new_data_label[:, -1], axis=0)]
        new_data_label = np.delete(new_data_label, -1, axis=1)
        print("mean res:", this_res.mean())

        # sorted_res = sorted(res.items(), key=lambda kv: [kv[1], kv[0]], reverse=True)
        #
        # print(sorted_res)
        #
        # res_data = []
        # for i in range(data_total):
        #     res_data.append(new_data[int(sorted_res[i][0])])

        res_data = new_data_label[:data_total]

        return res_data

    def gen_iter(self, iteration, rand_data, gen_num, v_range, sort):
        for i in range(iteration):
            self.data = self.fit_one_gen_iter(rand_data, gen_num, v_range, sort=sort)
            print("iteration:", str(i + 1), "finished")

        return self.data
